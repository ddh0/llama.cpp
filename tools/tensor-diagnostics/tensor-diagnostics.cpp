/**
 *
 * (name TBD): llama-tensor-diagnostics, llama-tensor-debug, llama-tensor-numerics ... ?
 *
 * This tool observes, prints info about, and exports to `.npy` ALL intermediate tensors (nodes)
 * during decoding (GGML compute graph execution). The model processes a single batch of tokens,
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

/* !! THIS FILE IS STILL UNDER ACTIVE DEVELOPMENT !! */

#include "llama-impl.h" // needed for LLAMA_LOG_* and llama_format_tensor_shape
#include "llama.h"
#include "common.h"
#include "arg.h"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <filesystem>


// elements with absolute values smaller than this are considered to be zero
static constexpr float ZERO_TOLERANCE = 0.005;


// utils for exporting GGML tensors as NumPy `.npy` files
// ref: https://numpy.org/doc/1.26/reference/generated/numpy.lib.format.html#format-version-1-0
namespace ggml_to_npy {

    // given a tensor name, return a file-safe name ending in .npy
    static std::string get_filename(const std::string & tensor_name) {
        std::string fname = tensor_name;
        for (char & c : fname) {
            if (std::string("/\\:*?\"<>| ").find(c) != std::string::npos) {
                c = '_';
            }
        }
        return fname + ".npy";
    }

    // given a GGML type, return the corresponding NumPy data type descriptor string
    // NOTE: this tool exports bf16 tensors in f32 for compatability, so bf16 is not allowed here.
    static std::string get_descr(ggml_type type) {
        switch (type) {
            case GGML_TYPE_F32:  return "'<f4'";
            case GGML_TYPE_F16:  return "'<f2'";
            case GGML_TYPE_I64:  return "'<i8'";
            case GGML_TYPE_I32:  return "'<i4'";
            case GGML_TYPE_I16:  return "'<i2'";
            case GGML_TYPE_I8:   return "'<i1'";
            case GGML_TYPE_BF16:
                throw std::runtime_error("BF16 is not currently supported for NumPy export "
                                         "(use f32 instead)");
            default:
                throw std::runtime_error(format(
                    "GGML type %s is not currently supported for NumPy export",
                    ggml_type_name(type)
                ));
        }
    }

    // write a minimal valid NumPy v1.0 header to the given file stream.
    // NOTE: the stream should be in binary mode.
    static void write_header(std::ofstream & out, const std::string & descr,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3)
    {
        out << "\x93NUMPY\x01\x00"; // magic string, major version, minor version (8 bytes total)

        // build header string
        std::string header = "{'descr': " + descr + ", 'fortran_order': False, 'shape': ("
            + std::to_string(ne3) + ", "  //
            + std::to_string(ne2) + ", "  // note the reversed ordering of tensor dimensions
            + std::to_string(ne1) + ", "  // between GGML and NumPy!
            + std::to_string(ne0) + ")}"; //

        // write header length (2 bytes) and contents (`header_len` bytes)
        const uint16_t header_len = static_cast<uint16_t>(header.length());
        out << reinterpret_cast<const char *>(&header_len) << header.c_str();

        const size_t n_bytes = 10 + header_len;
        const size_t n_pad = (64 - (n_bytes % 64)) % 64;

        if (n_pad > 0) {
            out << std::string(n_pad, ' ');
        }
    }

} // ggml_to_npy

struct session_stats_t {
    size_t n_capture = 0;              // incremented by 1 for every tensor that is observed by the callback
    size_t n_total_bytes_captured = 0; // combined size of all tensor data observed
    size_t n_zero_tensors = 0;         // total num. tensors observed whose elements were ALL zero
    size_t n_nan_tensors = 0;          // total num. tensors observed with one or more NaN elements
    size_t n_inf_tensors = 0;          // total num. tensors observed with one or more non-finite elements
    size_t n_zero_sized_tensors = 0;   // number of zero-sized tensors (they are skipped)
};

struct tensor_stats_t {
    size_t n_zeros = 0;
    size_t n_nans  = 0;
    size_t n_infs  = 0;
};

// process a single tensor and return stats.
// also update the session stats with the observed tensor stats.
static tensor_stats_t get_tensor_stats(session_stats_t * session_stats, ggml_tensor * t) {

    const int64_t n_elements = ggml_nelements(t);
    tensor_stats_t stats;

    if (n_elements > 0) {
        // valid GGML tensor dims start at 1 (i.e. the smallest valid shape is [1, 1, 1, 1])
        GGML_ASSERT(t->ne[0] > 0);
        GGML_ASSERT(t->ne[1] > 0);
        GGML_ASSERT(t->ne[2] > 0);
        GGML_ASSERT(t->ne[3] > 0);

        LLAMA_LOG_INFO("%s: %05zu: %-64s - [ %6lld, %6lld, %6lld, %6lld ], n_elements = %lld\n",
                       __func__, session_stats->n_capture, t->name,
                       t->ne[0], t->ne[1], t->ne[2], t->ne[3], n_elements);
    } else {
        ++session_stats->n_zero_sized_tensors;
        return stats; // return stats of all 0s
    }

    switch (t->type) {
        case GGML_TYPE_F32: {
            const float * f32_data = (const float *)t->data;
            for (int64_t i = 0; i < n_elements; ++i) {
                const float elem = f32_data[i];
                if (std::isnan(elem)) {
                    stats.n_nans++;
                }
                if (std::isinf(elem)) {
                    stats.n_infs++;
                }
                if (std::abs(elem) < ZERO_TOLERANCE) {
                    stats.n_zeros++;
                }
            }
            break;
        }
        case GGML_TYPE_F16: {
            const ggml_fp16_t * f16_data = (const ggml_fp16_t *)t->data;
            for (int64_t i = 0; i < n_elements; ++i) {
                const float elem = ggml_fp16_to_fp32(f16_data[i]);
                if (std::isnan(elem)) {
                    stats.n_nans++;
                }
                if (std::isinf(elem)) {
                    stats.n_infs++;
                }
                if (std::abs(elem) < ZERO_TOLERANCE) {
                    stats.n_zeros++;
                }
            }
            break;
        }
        case GGML_TYPE_BF16: {
            const ggml_bf16_t * bf16_data = (const ggml_bf16_t *)t->data;
            for (int64_t i = 0; i < n_elements; ++i) {
                const float elem = ggml_bf16_to_fp32(bf16_data[i]);
                if (std::isnan(elem)) {
                    stats.n_nans++;
                }
                if (std::isinf(elem)) {
                    stats.n_infs++;
                }
                if (std::abs(elem) < ZERO_TOLERANCE) {
                    stats.n_zeros++;
                }
            }
            break;
        }
        case GGML_TYPE_I64: {
            const int64_t * i64_data = (const int64_t *)t->data;
            for (int64_t i = 0; i < n_elements; ++i) {
                const int64_t elem = i64_data[i];
                if (std::isnan(elem)) {
                    stats.n_nans++;
                }
                if (std::isinf(elem)) {
                    stats.n_infs++;
                }
                if (std::abs(elem) < ZERO_TOLERANCE) {
                    stats.n_zeros++;
                }
            }
            break;
        }
        case GGML_TYPE_I32: {
            const int32_t * i32_data = (const int32_t *)t->data;
            for (int64_t i = 0; i < n_elements; ++i) {
                const int32_t elem = i32_data[i];
                if (std::isnan(elem)) {
                    stats.n_nans++;
                }
                if (std::isinf(elem)) {
                    stats.n_infs++;
                }
                if (std::abs(elem) < ZERO_TOLERANCE) {
                    stats.n_zeros++;
                }
            }
            break;
        }
        case GGML_TYPE_I16: {
            const int16_t * i16_data = (const int16_t *)t->data;
            for (int64_t i = 0; i < n_elements; ++i) {
                const int16_t elem = i16_data[i];
                if (std::isnan(elem)) {
                    stats.n_nans++;
                }
                if (std::isinf(elem)) {
                    stats.n_infs++;
                }
                if (std::abs(elem) < ZERO_TOLERANCE) {
                    stats.n_zeros++;
                }
            }
            break;
        }
        case GGML_TYPE_I8: {
            const int8_t * i8_data = (const int8_t *)t->data;
            for (int64_t i = 0; i < n_elements; ++i) {
                const int8_t elem = i8_data[i];
                if (std::isnan(elem)) {
                    stats.n_nans++;
                }
                if (std::isinf(elem)) {
                    stats.n_infs++;
                }
                if (std::abs(elem) < ZERO_TOLERANCE) {
                    stats.n_zeros++;
                }
            }
            break;
        }
        default:
            LLAMA_LOG_WARN("%s: tensor '%s' has unsupported type (%s) and will not be counted\n",
                           __func__, t->name, ggml_type_name(t->type));
            break;
    }

    ++session_stats->n_capture;
    session_stats->n_total_bytes_captured += ggml_nbytes(t);

    if (stats.n_zeros == static_cast<size_t>(n_elements)) {
        ++session_stats->n_zero_tensors;
    }
    if (stats.n_infs > 0) {
        ++session_stats->n_inf_tensors;
    }
    if (stats.n_nans > 0) {
        ++session_stats->n_nan_tensors;
    }

    return stats;
}

// callback function, receives tensors from scheduler
static bool tensor_diagnostic_cb(ggml_tensor * t, bool ask, void * user_data) {
    session_stats_t * session_stats = static_cast<session_stats_t *>(user_data);
    if (ask) {
        //
        // before graph compute, the scheduler asks us if we want to observe each tensor.
        // since this tool is primarily intended for diagnosing buggy or broken models,
        // rather than for productive inference, this tool will always observe all tensors.
        //
        return true;
    } else {
        const auto t_stats = get_tensor_stats(session_stats, t);
        // const auto t_fname = ggml_to_npy::get_filename(t->name);
        //
        // TODO
        //
        return true;
    }
}

// main diagnostic operation
static void run_diagnostics(llama_context * ctx, const common_params params) {

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
        LLAMA_LOG_WARN("%s: `--output-dir` parameter is not set; observed tensors will not be "
                       "saved!\n", __func__);
    }

    const auto n_batch = llama_n_batch(ctx);
    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(llama_get_model(ctx)));

    LLAMA_LOG_INFO("%s: tokenizing prompt ...\n", __func__);
    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, add_bos, false);
    const auto n_prompt_tokens = prompt_tokens.size();
    LLAMA_LOG_INFO("%s: n_prompt_tokens = %zu\n", __func__, n_prompt_tokens);

    if (n_prompt_tokens < n_batch) {
        LLAMA_LOG_WARN("%s: n_prompt_tokens %zu < n_batch %u; will pad with tok ID 0 "
                       "(this is most likely not a problem)\n", __func__, n_prompt_tokens, n_batch);
    }

    prompt_tokens.resize(n_batch);
    GGML_ASSERT(prompt_tokens.size() == n_batch);

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

    LLAMA_LOG_INFO("%s: running diagnostics over %d tokens ... this may take a while ...\n",
                   __func__, n_batch);

    auto ret = llama_decode(ctx, batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: llama_decode failed with code %d\n", __func__, ret);
    }

    LLAMA_LOG_INFO("%s: done.\n", __func__);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C"); // ref: https://github.com/ggml-org/llama.cpp/pull/17331

    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TENSOR_DIAGNOSTICS)) {
        return 1;
    }

    // enforce some basic parameters for now; maybe improve flexibility later
    const int32_t n_batch = std::max(params.n_ubatch, params.n_batch);
    params.n_ubatch   = n_batch;
    params.n_batch    = n_batch;
    params.n_ctx      = n_batch;
    params.warmup     = false;
    params.n_parallel = 1;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto session_stats = session_stats_t();

    params.cb_eval = tensor_diagnostic_cb;
    params.cb_eval_user_data = &session_stats;

    LLAMA_LOG_INFO("%s\n", common_params_get_system_info(params).c_str());
    common_init_result_ptr common_init = common_init_from_params(params);

    if (params.prompt.empty()) {
        LLAMA_LOG_ERROR("%s: no prompt provided via -f / --file\n", __func__);
        return 1;
    }

    llama_model * model = common_init->model();
    GGML_ASSERT(model != nullptr);

    llama_context * ctx = common_init->context();
    GGML_ASSERT(ctx != nullptr);

    run_diagnostics(ctx, params);

    // TODO: session_report(session_stats);

    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
