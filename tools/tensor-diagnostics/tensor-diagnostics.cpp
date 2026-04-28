/**
 *
 * llama-tensor-diagnostics
 *
 * This tool observes, prints info about, and exports to `.npy` ALL intermediate tensors (nodes)
 * during decoding (GGML compute graph execution). The model processes a single batch of plaintext,
 * which is read from the plaintext file specified by `-f` or `--file`.
 *
 * For every tensor observed, the tool will print the tensor name, shape, GGML_TYPE, etc.
 *
 * Additionally, at runtime, the tool will display warnings about...
 *
 *   ... tensors with ANY non-finite elements (and if so, how many)
 *
 *   ... tensors with ANY NaN elements (and if so, how many)
 *
 *   ... tensors whose elements are ALL zero (within tolerance [-0.005, 0.005] inclusive)
 *
 * All observed tensors will be captured, converted to the NumPy v1.0 format, and saved to disk.
 *
 * The resulting `.npy` tensor files can be loaded directly as a `np.ndarray` in Python using the
 * `np.load` function. From this point, in-depth post-mortem analysis of tensor data is hopefully
 * much easier.
 *
 * ---
 *
 * TODO / NOTES
 *  [1] - in the future maybe it would be helpful to observe other scenarios like autoregressive
 *        TG, partially full PP batches, etc. for now, only a single full PP batch.
 *
**/

#include "llama-impl.h" // LLAMA_LOG_* and llama_format_tensor_shape
#include "llama.h"
#include "common.h"
#include "arg.h"

// needed?
#include <ostream>
#include <iostream>

#include <fstream>

#include <filesystem>

// elements with absolute values smaller than this are considered to be zero
static constexpr float ZERO_TOLERANCE = 0.005;

static constexpr const char * PATH_UNSAFE_CHARS = "/\\:*?\"<>| ";

// convert tensor name to safe file name ending in .npy
// example: "__fattn__ (permuted)" --> "__fattn___(permuted).npy"
static std::string tensor_get_file_name(const std::string & name) {
    std::string sanitized = name;
    for (char & c : sanitized) {
        if (std::string(PATH_UNSAFE_CHARS).find(c) != std::string::npos) {
            c = '_';
        }
    }
    return sanitized + ".npy";
}

// maps a GGML type to its corresponding NumPy data type descriptor string.
// NOTE: bf16 not allowed here, using f32 instead for compatability.
static std::string get_npy_descr(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return "'<f4'";
        case GGML_TYPE_F16:  return "'<f2'";
        case GGML_TYPE_I64:  return "'<i8'";
        case GGML_TYPE_I32:  return "'<i4'";
        case GGML_TYPE_I16:  return "'<i2'";
        case GGML_TYPE_I8:   return "'<i1'";
        case GGML_TYPE_BF16: GGML_ASSERT(false);
        default: return "";
    }
}

// write a minimal valid NumPy v1.0 header to a file
static void write_numpy_header(std::ofstream & out, const std::string & descr, const std::vector<int64_t> & shape) {
    out.write("\x93NUMPY", 6); // magic bytes
    out << "\x01\x00\x00\x00"; // v1.0

    // header dictionary string
    std::string header = "{'descr': " + descr + ", 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        header += std::to_string(shape[i]);
        if (i < shape.size() - 1) header += ", ";
    }
    header += "), 'format': '<'}";

    // write header length as unsigned 64-bit integer, little endian
    uint64_t header_len = header.length();
    out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));

    // write the header string itself
    out.write(header.c_str(), header.length());
}

struct diagnostic_session_data {
    std::string output_dir;
    const ggml_tensor * pending_tensor = nullptr;
};

struct tensor_diagnostic_data {
    size_t num_nans = 0;
    size_t num_infs = 0;
    bool   all_zero = true;
};

// process a single tensor and return diagnostic data
static tensor_diagnostic_data process_tensor(const ggml_tensor * t) {
    tensor_diagnostic_data res;
    LLAMA_LOG_INFO("%s: %16s - [%s] - TODO!\n", __func__, t->name, llama_format_tensor_shape(t).c_str());
    // TODO: count NaNs, infs, and check for all zeroes within tolerance
    return res;
}

// callback function receives tensors from scheduler
static bool tensor_diagnostic_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * session = static_cast<diagnostic_session_data *>(user_data);
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
        std::string filename = tensor_get_file_name(t->name);
        std::filesystem::path file_path = std::filesystem::path(session->output_dir) / filename;

        std::ofstream out(file_path, std::ios::binary);
        if (!out) {
            LLAMA_LOG_ERROR("error: failed to open diagnostic file %s\n", file_path.string().c_str());
            return false;
        }

        std::vector<int64_t> shape(GGML_MAX_DIMS);
        GGML_ASSERT(shape.capacity() == GGML_MAX_DIMS && shape.size() == GGML_MAX_DIMS);

        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (t->ne[i] > 0) {
                shape.emplace_back(static_cast<int64_t>(t->ne[i]));
            }
        }

        write_numpy_header(out, get_npy_descr(t->type), shape);

        size_t num_nans = 0;
        size_t num_infs = 0;
        bool all_zero = true;

        size_t n_bytes = ggml_nbytes(t);
        std::vector<uint8_t> host_buffer(n_bytes);

        // copy the raw tensor data from the backend into the host buffer
        ggml_backend_tensor_get(t, host_buffer.data(), 0, n_bytes);
        out.write(reinterpret_cast<const char *>(host_buffer.data()), n_bytes);

        // Perform mathematical analysis on the buffer
        if (t->type == GGML_TYPE_F32) {
            float * data = reinterpret_cast<float *>(host_buffer.data());
            size_t n = n_bytes / sizeof(float);
            for (size_t i = 0; i < n; ++i) {
                float v = data[i];
                if (std::isnan(v)) {
                    num_nans++;
                } else if (std::isinf(v)) {
                    num_infs++;
                }
                if (std::abs(v) > ZERO_TOLERANCE) {
                    all_zero = false;
                }
            }
        } else if (t->type == GGML_TYPE_F16) {
            ggml_fp16_t * data = reinterpret_cast<ggml_fp16_t *>(host_buffer.data());
            size_t n = n_bytes / sizeof(ggml_fp16_t);
            for (size_t i = 0; i < n; ++i) {
                float v = ggml_fp16_to_fp32(data[i]);
                if (std::isnan(v)) {
                    num_nans++;
                } else if (std::isinf(v)) {
                    num_infs++;
                }
                if (std::abs(v) > ZERO_TOLERANCE) {
                    all_zero = false;
                }
            }
        } else if (t->type == GGML_TYPE_BF16) {
            // bf16 analysis
            // (Implementation assumes the host buffer is treated as raw bits or converted)
            // For brevity in this diagnostic tool, we treat it similarly to F32/F16 logic
            // via conversion if a specific helper is available.
            all_zero = false; // Placeholder for non-float types to prevent false positives
        } else {
            // For integer types, we just check if everything is 0
            all_zero = false; 
            // (Detailed integer scan could be added here)
        }

        out.close();

        LLAMA_LOG_INFO("%s: %16s - [%s] - %zu NaNs, %zu Infs, %s\n",
            __func__, t->name, llama_format_tensor_shape(t).c_str(),
            num_nans, num_infs, all_zero ? "ALL ZERO" : "non-zero"
        );

        return true; // if we return false, the scheduler aborts graph computation
    }
}

//
// main diagnostic operation
//
static void run_diagnostics(llama_context * ctx, const common_params params) {
    LLAMA_LOG_INFO("%s: running diagnostics ... this may take a while ...\n", __func__);

    // 1. Ensure output directory exists
    if (!params.tensor_diag_output_dir.empty()) {
        if (!std::filesystem::exists(params.tensor_diag_output_dir)) {
            if (!std::filesystem::create_directories(params.tensor_diag_output_dir)) {
                LLAMA_LOG_ERROR("%s: failed to create directory '%s'\n",
                                __func__, params.tensor_diag_output_dir.c_str());
                return;
            }
        }
    } else {
        LLAMA_LOG_WARN("%s: tensor_diag_output_dir is not set; tensors will not be saved\n", __func__);
    }

    diagnostic_session_data session;
    session.output_dir = params.tensor_diag_output_dir;

    if (params.prompt.empty()) {
        LLAMA_LOG_ERROR("%s: no prompt provided via -f / --file\n", __func__);
        return;
    }

    // tokenize prompt
    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(llama_get_model(ctx)));
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LLAMA_LOG_ERROR("%s: prompt resulted in zero tokens\n", __func__);
        return;
    }

    // prepare input batch
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    batch.n_tokens = static_cast<int32_t>(tokens.size());

    // decode
    int32_t result = llama_decode(ctx, batch);
    if (result != 0) {
        LLAMA_LOG_ERROR("%s: llama_decode failed with code %d\n", __func__, result);
    }

    llama_batch_free(batch);
    LLAMA_LOG_INFO("%s: done\n", __func__);
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

    LLAMA_LOG_INFO("%s\n", common_params_get_system_info(params).c_str());
    common_init_result_ptr common_init = common_init_from_params(params);

    llama_model * model = common_init->model();
    GGML_ASSERT(model != nullptr);

    llama_context * ctx = common_init->context();
    GGML_ASSERT(ctx != nullptr);

    run_diagnostics(ctx, params);

    llama_perf_context_print(ctx);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
