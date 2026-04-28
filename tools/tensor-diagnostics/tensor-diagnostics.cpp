/**
 *
 * llama-tensor-diagnostics
 *
 * This tool observes ALL intermediate tensors during graph execution, for a configurable number of
 * prompt processing and text generation batches. It will display info about...
 *
 *   ... tensors with ANY non-finite elements
 *
 *   ... tensors with ANY NaN elements
 *
 *   ... tensors whose elements are ALL zero (within tolerance [-0.005, 0.005] inclusive)
 *
 * All observed tensors will be captured, converted to the NumPy v1.0 format, and saved to disk.
 *
 * The resulting `.npy` tensor files can be loaded directly as a `np.ndarray` in Python using the 
 * `np.load` function. From this point, in-depth post-mortem analysis of tensor data is hopefully
 * much easier.
 *
**/

#include "log.h"
#include "llama.h"
#include "common.h"
#include "arg.h"

#include <filesystem>

//
// TRIAGE -- old / borrowed code that we might need, but is not yet integrated, needs to be cleaned
//
namespace triage {

    /**
     * Sanitizes a tensor name to be used as part of a filename.
     * Replaces characters that are invalid in filenames on common OSes with underscores.
     * @param name The original tensor name.
     * @return A sanitized string suitable for use in a filename.
     */
    static std::string sanitize_filename(const std::string & name) {
        std::string sanitized = name;
        for (char & c : sanitized) {
            if (std::string("/\\:*?\"<>|").find(c) != std::string::npos) {
                c = '_';
            }
        }
        return sanitized;
    }

    /**
     * Maps a GGML type to its corresponding NumPy data type descriptor string.
     * @param type The GGML data type.
     * @return A string representing the NumPy dtype, or an empty string if unsupported.
     */
    static std::string get_npy_descr(ggml_type type) {
        switch (type) {
            case GGML_TYPE_F32:  return "'<f4'";
            case GGML_TYPE_F16:  return "'<f2'";
            case GGML_TYPE_I64:  return "'<i8'";
            case GGML_TYPE_I32:  return "'<i4'";
            case GGML_TYPE_I16:  return "'<i2'";
            case GGML_TYPE_I8:   return "'<i1'";
            // Note: BF16 is handled by converting to F32 before this function is called.
            default: return "";
        }
    }

    /**
     * Callback to save a tensor's data to disk in NPY (NumPy) format v1.0.
     *
     * ref: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#format-version-1-0
     *
     * @param t current tensor
     * @param ask when ask is true, we return true if we want to receive the data for this tensor.
     * @param user_data a pointer to a `callback_data` struct.
     * @return always returns true to continue graph execution.
     */
    static bool save_tensor_to_npy(struct ggml_tensor * t, bool ask, void * user_data) {
        if (ask) {
            // currently only support non-quantized tensors, which we can easily save to NPY format
            if (ggml_is_quantized(t->type)) {
                return false;
            }
            ggml_type type_to_check = t->type;
            if (type_to_check == GGML_TYPE_BF16) {
                type_to_check = GGML_TYPE_F32; // we promise to convert BF16 GGML tensors to F32 GGML tensors before converting/writing
            }
            return !get_npy_descr(type_to_check).empty();
        }

        auto * cb_data = (callback_data *) user_data;

        // prepare tensor data (ensure it's contiguous and in a supported format)

        ggml_type type_to_save = t->type;
        std::vector<uint8_t> data_to_save;

        if (t->type == GGML_TYPE_BF16) {
            // we are going to convert BF16 data to F32 
            type_to_save = GGML_TYPE_F32;
            const int64_t n_elems = ggml_nelements(t);
            data_to_save.resize(n_elems * sizeof(float));

            // create a temporary buffer for the bf16 data
            std::vector<uint8_t> bf16_data(ggml_nbytes(t));
            ggml_backend_tensor_get(t, bf16_data.data(), 0, ggml_nbytes(t));

            // manually convert bf16 to f32
            float * dst_f32 = (float *) data_to_save.data();
            ggml_bf16_t * src_bf16 = (ggml_bf16_t *) bf16_data.data();
            for (int64_t i = 0; i < n_elems; ++i) {
                dst_f32[i] = ggml_bf16_to_fp32(src_bf16[i]);
            }
        } else {
            // for other types, get a contiguous copy from the backend.
            // this handles GPU --> CPU transfers and non-contiguous tensors automatically.
            data_to_save.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, data_to_save.data(), 0, ggml_nbytes(t));
        }

        // construct file header string

        std::string descr = get_npy_descr(type_to_save);
        std::string shape_str = "(";
        const int n_dims = ggml_n_dims(t);
        if (n_dims > 0) {
            for (int i = n_dims - 1; i >= 0; --i) {
                shape_str += std::to_string(t->ne[i]);
                if ((n_dims > 1 && i > 0)) {
                    shape_str += ", ";
                }
            }
        }
        if (n_dims == 1) {
            shape_str += ",";
        }
        shape_str += ")";

        std::string header_dict_nl = "{'descr': " + descr + ", 'fortran_order': False, 'shape': " + shape_str + ", }\n";

        // determine filename

        std::string descriptive_name;
        auto it = cb_data->tensor_descriptive_names.find(t);
        if (it != cb_data->tensor_descriptive_names.end()) {
            descriptive_name = it->second;
        } else {
            descriptive_name = t->name;
        }

        if (descriptive_name.empty()) {
            descriptive_name = "unnamed";
        }

        // create file

        LOG("%s: saving tensor '%s'\n", __func__, descriptive_name.c_str());
        LOG("%s: -- op: %s, type: %s, shape: [%s]\n", __func__, ggml_op_desc(t), ggml_type_name(t->type), ggml_ne_string(t).c_str());

        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << cb_data->file_counter++ << "_" << sanitize_filename(descriptive_name) << ".npy";
        std::string filename = ss.str();

        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            LOG_ERR("%s: -- failed to open file '%s' for writing\n", __func__, filename.c_str());
            return true;
        }

        // write file header

        // magic string and version 1.0
        file.write("\x93NUMPY", 6);
        file.put('\x01');
        file.put('\x00');

        // header length and padding calculation
        size_t unpadded_len = 10 + header_dict_nl.length(); // magic, version, and length fields
        size_t padding = (64 - (unpadded_len % 64)) % 64;
        std::string header_padded = header_dict_nl;
        header_padded.insert(header_padded.length() - 1, padding, ' ');
        uint16_t header_len_val = header_padded.length();

        // write header length (2 bytes, LE)
        file.put(header_len_val & 0xFF);
        file.put((header_len_val >> 8) & 0xFF);

        // write header content
        file.write(header_padded.c_str(), header_padded.length());

        // write tensor data
        file.write(reinterpret_cast<const char*>(data_to_save.data()), data_to_save.size());
        file.close();
        LOG("%s: -- saved to %s\n", __func__, filename.c_str());
        return true;
    }
} // triage

struct tensor_diagnostic_data {
    size_t num_nans = 0;
    size_t num_infs = 0;
    bool   all_zero = true;
};

// process a single tensor and return diagnostic data
static tensor_diagnostic_data process_tensor(const ggml_tensor * t) {
    tensor_diagnostic_data res;
    // TODO: count NaNs, infs, and check for all zeroes within tolerance
    return res;
}

// callback function receives tensors from scheduler
static bool tensor_diagnostic_cb(ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        //
        // before graph compute, the scheduler asks us if we want to observe each tensor.
        // since this tool is primarily intended for diagnosing buggy or broken models (and not for
        // general-purpose use by end-users), it is probably preferable to ALWAYS observe ALL
        // tensors that we possibly can.
        //
        // in the future, we can almost certainly safely ignore certain types of tensors,
        // particularly permutations and reshapings of tensors that we _did not_ ignore. currently,
        // while this tool is under development, we will signal that we want to observe all tensors.
        //
        return true;
    } else {
        // we are in-flight; the scheduler is passing us the real tensor for observation.

        auto * cb_data = (tensor_diagnostic_data *) user_data;
        return true; // if we return false, the scheduler aborts graph computation
    }
};

//
// main diagnostic operation:
//
//  1. read text from the file at the path specified by `text_input_file` as input to the model
//  2. during the decoding in step 1, capture every intermediate tensor (all nodes in the graph)
//     and pass them
//
// results are saved under the directory specified by `diag_output_dir`, in subdirectories:
//  - `./pp/`: stores the tensors captured during prompt processing
//  - `./tg/`: stores the tensors captured during text generation
//
// if the directory specified by `diag_output_dir` already exists, the program will abort to prevent
// accidentally overwriting other diagnostic dumps.
//
static void run_diagnostics(llama_context * ctx, const common_params params) {
    LOG_INF("%s: running diagnostics ... this may take a while ...\n", __func__);

    // ... TODO ...

    LOG_INF("%s: done\n", __func__);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TENSOR_DIAGNOSTICS)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto cb_data = tensor_diagnostic_data();

    params.cb_eval = tensor_diagnostic_cb;
    params.cb_eval_user_data = &cb_data;

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    common_init_result_ptr common_init = common_init_from_params(params);

    llama_model * model = common_init->model(); GGML_ASSERT(model != nullptr);
    llama_context * ctx = common_init->context(); GGML_ASSERT(ctx != nullptr);

    run_diagnostics(ctx, params);

    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
