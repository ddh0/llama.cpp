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

#include "llama-impl.h" // needed for LLAMA_LOG_* and llama_format_tensor_shape
#include "llama.h"
#include "common.h"
#include "arg.h"

#include <cmath>
#include <fstream>
#include <filesystem>

// elements with absolute values smaller than this are considered to be zero
static constexpr float ZERO_TOLERANCE = 0.005;

// utils for exporting GGML tensor data into NumPy v1.0 format
namespace ggml_to_npy {

    // convert tensor name to safe file name ending in .npy
    // example: "__fattn__ (permuted)" --> "__fattn___(permuted).npy"
    static std::string get_filename(const std::string & name) {
        std::string sanitized = name;
        for (char & c : sanitized) {
            if (std::string("/\\:*?\"<>| ").find(c) != std::string::npos) {
                c = '_';
            }
        }
        return sanitized + ".npy";
    }

    // maps a GGML type to its corresponding NumPy data type descriptor string.
    // NOTE: this tool exports bf16 tensors in f32 for compatability, so bf16 is not allowed here.
    static std::string get_desc(ggml_type type) {
        switch (type) {
            case GGML_TYPE_F32:  return "'<f4'";
            case GGML_TYPE_F16:  return "'<f2'";
            case GGML_TYPE_I64:  return "'<i8'";
            case GGML_TYPE_I32:  return "'<i4'";
            case GGML_TYPE_I16:  return "'<i2'";
            case GGML_TYPE_I8:   return "'<i1'";
            case GGML_TYPE_BF16:
                throw std::runtime_error("bf16 numpy export is not supported, use f32");
            default: return "";
        }
    }

    // write a minimal valid NumPy v1.0 header to the given file stream
    static void write_header(std::ofstream & out,
        const std::string & descr,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3)
    {
        // valid tensors dims start at 1
        GGML_ASSERT(ne0 > 0);
        GGML_ASSERT(ne1 > 0);
        GGML_ASSERT(ne2 > 0);
        GGML_ASSERT(ne3 > 0);

        out.write("\x93NUMPY", 6); // start header
        out << "\x01\x00\x00\x00"; // v1.0

        // header dictionary string
        std::string header = "{'descr': " + descr + ", 'fortran_order': False, 'shape': (";
        header += std::to_string(ne0) + ", " + std::to_string(ne1) + ", " +
                  std::to_string(ne2) + ", " + std::to_string(ne3) + "), 'format': '<'}";

        // write header length (little endian)
        uint16_t header_len = static_cast<uint16_t>(header.length());
        out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));

        // write header string
        out.write(header.c_str(), header.length());
    }

} // ggml_to_npy

struct all_stats {
    size_t n_zeros = 0;
    size_t n_nans  = 0;
    size_t n_infs  = 0;
};

struct per_tensor_stats {
    size_t n_zeros = 0;
    size_t n_nans  = 0;
    size_t n_infs  = 0;
};

// process a single tensor and return stats
static per_tensor_stats get_tensor_stats(const ggml_tensor * t) {
    GGML_ASSERT(t != nullptr);

    per_tensor_stats stats;
    int64_t n_elem = ggml_nelements(t);

    if (n_elem == 0) {
        return stats;
    }

    const uint8_t * data = (const uint8_t *)t->data;
    size_t n_elements = static_cast<size_t>(n_elem);

    switch (t->type) {
        case GGML_TYPE_F32: {
            const float * f32_data = (const float *)data;
            for (size_t i = 0; i < n_elements; ++i) {
                float val = f32_data[i];
                if (std::isnan(val)) stats.n_nans++;
                if (std::isinf(val)) stats.n_infs++;
                if (std::abs(val) <= ZERO_TOLERANCE) stats.n_zeros++;
            }
            break;
        }
        case GGML_TYPE_F16: {
            const ggml_fp16_t * f16_data = (const ggml_fp16_t *)data;
            for (size_t i = 0; i < n_elements; ++i) {
                float val = ggml_fp16_to_fp32(f16_data[i]);
                if (std::isnan(val)) stats.n_nans++;
                if (std::isinf(val)) stats.n_infs++;
                if (std::abs(val) <= ZERO_TOLERANCE) stats.n_zeros++;
            }
            break;
        }
        default:
            LLAMA_LOG_WARN("%s: unsupported type\n", __func__);
            break;
    }

    return stats;
}

// callback function receives tensors from scheduler
static bool tensor_diagnostic_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * session_stats = static_cast<all_stats *>(user_data);
    if (ask) {
        //
        // before graph compute, the scheduler asks us if we want to observe each tensor.
        // since this tool is primarily intended for diagnosing buggy or broken models, rather than
        // for actual inference, it is probably preferable to ALWAYS observe ALL tensors that we
        // possibly can.
        //
        // in the future, we can almost certainly safely ignore certain types of tensors,
        // particularly permutations and reshapings of tensors that we _did not_ ignore. currently,
        // while this tool is under development, we will signal that we want to observe all tensors.
        //
        return true;
    } else {
        per_tensor_stats t_stats = get_tensor_stats(t);
        session_stats->n_zeros += t_stats.n_zeros;
        session_stats->n_nans  += t_stats.n_nans;
        session_stats->n_infs  += t_stats.n_infs;
        return true;
    }
}

//
// main diagnostic operation
//
static void run_diagnostics(llama_context * ctx, const common_params params) {
    LLAMA_LOG_INFO("%s: running diagnostics ... this may take a while ...\n", __func__);

    // ensure output directory exists
    if (!params.tensor_diag_output_dir.empty()) {
        if (!std::filesystem::exists(params.tensor_diag_output_dir)) {
            if (!std::filesystem::create_directories(params.tensor_diag_output_dir)) {
                LLAMA_LOG_ERROR("%s: failed to create directory at %s\n",
                                __func__, params.tensor_diag_output_dir.c_str());
                return;
            }
        }
    } else {
        LLAMA_LOG_WARN("%s: `--output-dir` parameter is not set; observed tensors will not be saved!\n",
                       __func__);
    }

    if (params.prompt.empty()) {
        LLAMA_LOG_ERROR("%s: no prompt provided via -f / --file\n", __func__);
        return;
    }

    const auto n_batch = llama_n_batch(ctx);
    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(llama_get_model(ctx)));

    // tokenize prompt
    std::vector<llama_token> prompt_tokens =
        common_tokenize(ctx, params.prompt, add_bos, /* parse_special = */ false);

    const auto n_prompt_tokens = prompt_tokens.size();

    if (n_prompt_tokens == 0) {
        LLAMA_LOG_ERROR("%s: no tokens in prompt; cannot proceed\n", __func__);
        return;
    }
    if (n_prompt_tokens < n_batch) {
        LLAMA_LOG_WARN("%s: n_prompt_tokens %zu < n_batch %u; will pad with tok ID 0 "
                       "(this is most likely not a problem)\n", __func__, n_prompt_tokens, n_batch);

    }

    // this truncates prompt to one full batch if too large, or pads with 0s if too small
    prompt_tokens.resize(n_batch);
    GGML_ASSERT(prompt_tokens.size() == n_batch && n_prompt_tokens == n_batch);

    // note that we only fill and decode one sequence for now.
    // in the future this can be configurable
    llama_batch batch = llama_batch_init(static_cast<int32_t>(n_batch), 0, 1);
    for (size_t i = 0; i < n_batch; ++i) {
        batch.token[i] = prompt_tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    batch.n_tokens = n_batch;

    int32_t res = llama_decode(ctx, batch);
    if (res != 0) {
        LLAMA_LOG_ERROR("%s: llama_decode failed with code %d\n", __func__, res);
    }

    LLAMA_LOG_INFO("%s: done\n", __func__);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C"); // ref: https://github.com/ggml-org/llama.cpp/pull/17331

    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TENSOR_DIAGNOSTICS)) {
        return 1;
    }

    // enforce some basic parameters for now; maybe improve flexibility later
    params.n_ubatch = params.n_batch;
    params.n_ctx = params.n_batch;
    params.warmup = false;
    params.n_parallel = 1;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto cb_data = all_stats();

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

    llama_backend_free();

    return 0;
}
