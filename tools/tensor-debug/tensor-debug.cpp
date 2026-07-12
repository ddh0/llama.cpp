/**
 *
 * llama-tensor-debug
 *
 * This tool observes, prints info about, and optionally exports ALL intermediate tensors (nodes)
 * during inference (compute graph execution). The model processes a single batch of tokens, which
 * can be specified with `-p / --prompt` or `-f / --file`.
 *
 * For every tensor observed, the tool will print the name, shape, GGML_TYPE, etc.
 *
 * Additionally, at runtime, the tool will display warnings about...
 *
 *   ... tensors with ANY non-finite elements (and if so, how many)
 *
 *   ... tensors with ANY NaN elements (and if so, how many)
 *
 *   ... tensors whose elements are ALL strictly smaller than 0.005 in absolute value
 *       (all-zero tensors)
 *
 * All observed tensors will be captured, converted to the NumPy v1.0 format, and saved to disk.
 *
 * The resulting `.npy` tensor files can be loaded directly as a `np.ndarray` in Python using the
 * `np.load` function. From that point, in-depth analysis of tensor data is hopefully much easier.
 *
 * ---
 *
 * TODO / NOTES
 *  [1] - in the future it may be helpful to observe other scenarios like autoregressive text gen,
 *        various batch sizes, etc.
 *
**/

#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>
#include <cinttypes>
#include <filesystem>

#define TDIM "%7" PRId64 // tensor dimension format string
#define TNAME "%-54s"    // tensor name format string

// elements with absolute values strictly smaller than this are considered to be zero
static constexpr float ZERO_TOLERANCE = 0.005;

// enforce a small batch size and context size to avoid unnecessary disk usage and runtime
static constexpr int32_t MAX_N_BATCH = 512;
static constexpr int32_t MAX_N_CTX = 8192;


// helper struct to access tensor data regardless of backend
struct tensor_data_view {
    tensor_data_view(void * ptr) : ptr_(ptr), buf_({}) {}
    tensor_data_view(std::vector<uint8_t>&& buf) : ptr_(buf.data()), buf_(std::move(buf)) {}
    inline void * data() const { return ptr_; } // return a pointer to the tensor data
    private:
        void * ptr_;
        std::vector<uint8_t> buf_;
};

// helper function to access tensor data regardless of backend
static tensor_data_view tensor_get_view(const ggml_tensor * t) {
    if (ggml_backend_buffer_is_host(t->buffer)) { // XXX: is this really correct?
        return tensor_data_view(t->data);
    }
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> buf(nbytes);
    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
    return tensor_data_view(std::move(buf));
}


// utils for exporting GGML tensors as NumPy `.npy` files
// ref: https://numpy.org/doc/1.26/reference/generated/numpy.lib.format.html#format-version-1-0
namespace ggml_to_npy {

    // given a tensor name, return a file-safe name ending in .npy
    static std::string get_fname(const std::string & tensor_name) {
        std::string fname = tensor_name;
        for (char & c : fname) {
            if (std::string("/\\:*?\"<>| ").find(c) != std::string::npos) {
                c = '_';
            }
        }
        return fname + ".npy";
    }

    // given a GGML type, return the corresponding NPY type descriptor string
    //
    // NOTE: this function returns the F32 descriptor for BF16 tensors.
    //       it is the caller's responsibility to convert BF16 -> F32 before writing.
    static std::string get_descr(ggml_type src_type) {
        switch (src_type) {
            case GGML_TYPE_F32:  return "'<f4'";
            case GGML_TYPE_BF16: return "'<f4'"; // will upcast
            case GGML_TYPE_F16:  return "'<f2'";
            case GGML_TYPE_I64:  return "'<i8'";
            case GGML_TYPE_I32:  return "'<i4'";
            case GGML_TYPE_I16:  return "'<i2'";
            case GGML_TYPE_I8:   return "'<i1'";
            default:
                GGML_ABORT("%s: GGML type %s is not supported for NumPy export",
                           __func__, ggml_type_name(src_type));
        }
    }

    // write a minimal valid NumPy v1.0 header to the given file stream.
    // called from inside `save_tensor`; user should not need to call this directly.
    //
    // NOTE: it is the caller's responsibility to ensure that `fout` was opened in binary mode.
    static void write_header(std::ofstream & fout, const std::string & descr,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3)
    {
        // magic string, major version, minor version
        fout.write("\x93NUMPY\x01\x00", 8);

        std::string header = "{'descr': " + descr + ", 'fortran_order': False, 'shape': ("
            + std::to_string(ne3) + ", "
            + std::to_string(ne2) + ", "
            + std::to_string(ne1) + ", "
            + std::to_string(ne0) + ")}";

        const auto header_str_len = static_cast<uint16_t>(header.length());
        fout.write(reinterpret_cast<const char *>(&header_str_len), sizeof(header_str_len));
        fout.write(header.c_str(), header_str_len);

        const size_t n_bytes_header = 10 + header_str_len;
        const size_t n_pad = (64 - (n_bytes_header % 64)) % 64;

        if (n_pad > 0) {
            fout << std::string(n_pad, ' ');
        }
    }

    // NOTE: this function assumes the tensor is already validated (i.e. not nullptr and not empty)
    //
    // TODO: avoid potentially copying the same tensor data twice: change the signature to accept:
    //        - ggml_type src_type
    //        -    size_t n_elem
    //        -    void * t_data
    //
    static void save_tensor(const std::string & fpath, const ggml_tensor * t) {
        const ggml_type src_type = t->type;
        const auto n_elem = static_cast<size_t>(ggml_nelements(t));

        std::ofstream out(fpath, std::ios::binary);
        if (!out.is_open()) {
            LOG_ERR("%s: failed to open file for writing: %s\n", __func__, fpath.c_str());
            return;
        }

        write_header(out, get_descr(src_type), t->ne[0], t->ne[1], t->ne[2], t->ne[3]);

        const void * t_data = tensor_get_view(t).data();

        // this switch must logically agree with `get_descr()`
        switch (src_type) {
            case GGML_TYPE_F32: {
                out.write(reinterpret_cast<const char *>(t_data), n_elem * sizeof(float));
            } break;
            case GGML_TYPE_BF16: {
                std::vector<float> buf(n_elem);
                auto data = static_cast<const ggml_bf16_t *>(t_data);
                for (size_t i = 0; i < n_elem; ++i) {
                    buf[i] = ggml_bf16_to_fp32(data[i]);
                }
                out.write(reinterpret_cast<const char *>(buf.data()), n_elem * sizeof(float));
            } break;
            case GGML_TYPE_F16: {
                auto data = static_cast<const ggml_fp16_t *>(t_data);
                out.write(reinterpret_cast<const char *>(data), n_elem * sizeof(ggml_fp16_t));
            } break;
            case GGML_TYPE_I64: {
                auto data = static_cast<const int64_t *>(t_data);
                out.write(reinterpret_cast<const char *>(data), n_elem * sizeof(int64_t));
            } break;
            case GGML_TYPE_I32: {
                auto data = static_cast<const int32_t *>(t_data);
                out.write(reinterpret_cast<const char *>(data), n_elem * sizeof(int32_t));
            } break;
            case GGML_TYPE_I16: {
                auto data = static_cast<const int16_t *>(t_data);
                out.write(reinterpret_cast<const char *>(data), n_elem * sizeof(int16_t));
            } break;
            case GGML_TYPE_I8: {
                auto data = static_cast<const int8_t *>(t_data);
                out.write(reinterpret_cast<const char *>(data), n_elem * sizeof(int8_t));
            } break;
            default:
                GGML_ABORT("%s: GGML type %s is not supported for NumPy export",
                           __func__, ggml_type_name(src_type));
        }
    }
} // ggml_to_npy


// session data
struct tensor_debug_cb_data {
    size_t tensor_idx       = 0; // incremented by 1 for every tensor that is observed
    size_t n_zero_tensors   = 0; // number of tensors observed whose elements were ALL zero
    size_t n_nan_tensors    = 0; // number of tensors observed with one or more NaN elements
    size_t n_inf_tensors    = 0; // number of tensors observed with one or more non-finite elements
    size_t n_bytes_captured = 0; // combined size of all tensor data captured
};

// per-tensor statistics
struct tensor_stats_t {
    size_t n_zeros = 0;
    size_t n_nans  = 0;
    size_t n_infs  = 0;
    size_t n_elem  = 0;

    float min_val = INFINITY;
    float max_val = -INFINITY;

    // stats for Welford's Algorithm. ref:
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    double mean = 0.0;
    double m2   = 0.0;

    // this flag is set for tensors whose types are not (currently) supported by this tool
    bool ignore = false;

    void read_flt(float elem) {
        ++n_elem;
        if (std::isnan(elem)) {
            ++n_nans;
            return;
        }
        if (std::isinf(elem)) {
            ++n_infs;
        }
        if (std::abs(elem) < ZERO_TOLERANCE) {
            ++n_zeros;
        }

        min_val = std::min(elem, min_val);
        max_val = std::max(elem, max_val);

        // accumulate per-tensor mean and variance from this element
        double delta = elem - mean;
        mean += delta / n_elem;
        double delta2 = elem - mean;
        m2 += delta * delta2;
    }

    template <typename T>
    void read_int(T elem) {
        static_assert(std::is_integral_v<T>, "read_int expects an integral type");
        ++n_elem;
        if (elem == 0) {
            ++n_zeros;
        }

        const auto elemf = static_cast<float>(elem); // cast to float for computing stats

        min_val = std::min(elemf, min_val);
        max_val = std::max(elemf, max_val);

        // accumulate per-tensor mean and variance from this element
        double delta = elemf - mean;
        mean += delta / n_elem;
        double delta2 = elemf - mean;
        m2 += delta * delta2;
    }

    double get_variance() const {
        if (n_elem < 2) {
            return 0.0;
        }
        return m2 / n_elem;
    }
};

// process a single tensor and return stats.
//
// NOTE: this function assumes the tensor is already validated (i.e. not nullptr and not empty)
static tensor_stats_t get_tensor_stats(const ggml_tensor * t) {
    tensor_stats_t stats;
    const auto n_elements = static_cast<size_t>(ggml_nelements(t));
    const void * t_data = tensor_get_view(t).data();
    switch (t->type) {
        case GGML_TYPE_F32: {
            auto f32_data = static_cast<const float *>(t_data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_flt(f32_data[i]);
            }
        } break;
        case GGML_TYPE_F16: {
            auto f16_data = static_cast<const ggml_fp16_t *>(t_data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_flt(ggml_fp16_to_fp32(f16_data[i]));
            }
        } break;
        case GGML_TYPE_BF16: {
            auto bf16_data = static_cast<const ggml_bf16_t *>(t_data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_flt(ggml_bf16_to_fp32(bf16_data[i]));
            }
        } break;
        case GGML_TYPE_I64: {
            auto i64_data = static_cast<const int64_t *>(t_data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_int(i64_data[i]);
            }
        } break;
        case GGML_TYPE_I32: {
            auto i32_data = static_cast<const int32_t *>(t_data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_int(i32_data[i]);
            }
        } break;
        case GGML_TYPE_I16: {
            auto i16_data = static_cast<const int16_t *>(t_data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_int(i16_data[i]);
            }
        } break;
        case GGML_TYPE_I8: {
            auto i8_data = static_cast<const int8_t *>(t_data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_int(i8_data[i]);
            }
        } break;
        default:
            LOG_WRN("%s: tensor '" TNAME "' has unsupported type (%s), will be ignored\n",
                    __func__, t->name, ggml_type_name(t->type));
            stats.ignore = true;
            return stats;
    }

    GGML_ASSERT(stats.n_elem == n_elements); // sanity check
    return stats;
}

// callback function, receives tensors from scheduler
static bool tensor_debug_cb(ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        GGML_ASSERT(t != nullptr && "callback received tensor == nullptr!");
        //
        // before graph compute, the scheduler asks us if we want to observe each tensor.
        // since this tool is primarily intended for diagnosing buggy or broken models,
        // rather than for productive inference, this tool will always observe all
        // tensors which are not empty.
        //
        // TODO: - add CLI params to optionally also filter (permuted) and (view) tensors
        //       - warn about tensors that have crazy min/avg/max/var
        //       - add CLI params to only save tensors that are warned about to disk
        //
        return !ggml_is_empty(t);
    } else {
        GGML_ASSERT(t != nullptr && "callback received tensor == nullptr!");
        GGML_ASSERT(!ggml_is_empty(t) && "callback received empty tensor! (should be filtered)");

        const auto t_stats = get_tensor_stats(t);

        if (t_stats.ignore) {
            return true;
        }

        const bool t_all_zero = t_stats.n_zeros == t_stats.n_elem;

        auto cb_data = static_cast<tensor_debug_cb_data *>(user_data);
        ++cb_data->tensor_idx;
        cb_data->n_bytes_captured += ggml_nbytes(t);

        if (t_all_zero) {
            ++cb_data->n_zero_tensors;
        }
        if (t_stats.n_infs > 0) {
            ++cb_data->n_inf_tensors;
        }
        if (t_stats.n_nans > 0) {
            ++cb_data->n_nan_tensors;
        }

        if (t_all_zero || t_stats.n_infs > 0 || t_stats.n_nans > 0) {
            LOG_WRN(
                "%05zu: %4s: " TNAME " [ " TDIM ", " TDIM ", " TDIM ", " TDIM " ] all zero = %1s, "
                "infs = %7zu, nans = %5zu, min = %9.2f, avg = %6.2f, max = %9.2f, var = %9.2f\n",
                cb_data->tensor_idx, ggml_type_name(t->type), t->name, t->ne[0], t->ne[1], t->ne[2],
                t->ne[3], t_all_zero ? "1" : "0", t_stats.n_infs, t_stats.n_nans, t_stats.min_val,
                t_stats.mean, t_stats.max_val, t_stats.get_variance());
        } else {
            LOG_INF(
                "%05zu: %4s: " TNAME " [ " TDIM ", " TDIM ", " TDIM ", " TDIM " ] all zero = %1s, "
                "infs = %7zu, nans = %5zu, min = %9.2f, avg = %6.2f, max = %9.2f, var = %9.2f\n",
                cb_data->tensor_idx, ggml_type_name(t->type), t->name, t->ne[0], t->ne[1], t->ne[2],
                t->ne[3], t_all_zero ? "1" : "0", t_stats.n_infs, t_stats.n_nans, t_stats.min_val,
                t_stats.mean, t_stats.max_val, t_stats.get_variance());
        }

        // TODO: ggml_to_npy::save_tensor(fpath, t)
        return true;
    }
}

static void print_report(void * user_data) {
    auto cb_data = static_cast<tensor_debug_cb_data *>(user_data);
    LOG_INF("%s: n_zero_tensors: %6zu\n", __func__, cb_data->n_zero_tensors);
    LOG_INF("%s:  n_inf_tensors: %6zu\n", __func__, cb_data->n_inf_tensors);
    LOG_INF("%s:  n_nan_tensors: %6zu\n", __func__, cb_data->n_nan_tensors);
    LOG_INF("%s:      n_tensors: %6zu\n", __func__, cb_data->tensor_idx);
    LOG_INF("%s: total tensor data captured: %8.2f MiB\n",
            __func__, cb_data->n_bytes_captured/1024.0/1024.0);
}

//
// debug driver function
//
//  - tokenize prompt, truncate to n_batch
//  - fill a single batch of tokens
//  - evaluation of the batch by `llama_decode` triggers the callback
//
static void run_debug(llama_context * ctx, const std::string & prompt) {

    const auto n_seq_max = static_cast<int32_t>(llama_n_seq_max(ctx));
    const auto n_batch = static_cast<int32_t>(llama_n_batch(ctx));

    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(llama_get_model(ctx)));
    LOG_INF("%s: tokenizing prompt ...\n", __func__);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, prompt, add_bos, false);
    const auto n_prompt_tokens = static_cast<int32_t>(prompt_tokens.size());

    if (n_prompt_tokens < n_batch) {
        LOG_INF("%s: n_prompt_tokens (%d) < n_batch (%d); will pad with tok ID 0\n",
                       __func__, n_prompt_tokens, n_batch);
    } else {
        LOG_INF("%s: n_prompt_tokens = %d\n", __func__, n_prompt_tokens);
    }

    prompt_tokens.resize(n_batch);
    GGML_ASSERT(static_cast<int32_t>(prompt_tokens.size()) == n_batch);
    GGML_ASSERT(n_seq_max > 0);

    llama_batch batch = llama_batch_init(/* n_tokens */ n_batch, /* embd */ 0, /* n_seq_max */ 1);
    for (int32_t i = 0; i < n_batch; ++i) {
        batch.token[i] = prompt_tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    batch.n_tokens = n_batch;

    LOG_INF("%s: decoding %d tokens for debugging ... this may take a while ...\n",
                   __func__, batch.n_tokens);

    auto ret = llama_decode(ctx, batch);
    if (ret != 0) {
        LOG_ERR("%s: llama_decode failed with code %d\n", __func__, ret);
    }

    llama_batch_free(batch);

    LOG_INF("%s: done.\n", __func__);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C"); // ref: https://github.com/ggml-org/llama.cpp/pull/17331

    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TENSOR_DEBUG)) {
        return EXIT_FAILURE;
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s: no prompt provided; please specify `-p / --prompt` or `-f / --file`\n",
                        __func__);
        return EXIT_FAILURE;
    }

    // enforce a small batch size to avoid unnecessary disk usage and runtime
    const int32_t n_batch = std::min(std::min(params.n_ubatch, params.n_batch), MAX_N_BATCH);
    if (params.n_ubatch != n_batch) {
        LOG_WRN("%s: parameter override: n_ubatch: %d -> %d\n", __func__, params.n_ubatch, n_batch);
        params.n_ubatch = n_batch;
    }
    if (params.n_batch != n_batch) {
        LOG_WRN("%s: parameter override: n_batch: %d -> %d\n", __func__, params.n_batch, n_batch);
        params.n_batch = n_batch;
    }

    // enforce a small context size to avoid unnecessary disk usage and runtime
    if (params.n_ctx > MAX_N_CTX) {
        LOG_WRN("%s: parameter override: n_ctx: %d -> %d\n", __func__, params.n_ctx, MAX_N_CTX);
        params.n_ctx = MAX_N_CTX;
    }

    // TODO: support multiple sequences?
    if (params.n_parallel != 1) {
        LOG_WRN("%s: parameter override: n_parallel: %d -> 1\n", __func__, params.n_parallel);
        params.n_parallel = 1;
    }

    // the callback would capture the empty run
    if (params.warmup) {
        LOG_WRN("%s: parameter override: warmup = false\n", __func__);
        params.warmup = false;
    }

    auto cb_data = tensor_debug_cb_data();

    params.cb_eval = tensor_debug_cb;
    params.cb_eval_user_data = &cb_data;

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    llama_backend_init();
    llama_numa_init(params.numa);
    common_init_result_ptr common_init = common_init_from_params(params);

    llama_model * model = common_init->model();
    if (model == nullptr) {
        return EXIT_FAILURE;
    }

    llama_context * ctx = common_init->context();
    if (ctx == nullptr) {
        return EXIT_FAILURE;
    }

    // ensure output directory exists
    if (!params.tensor_dbg_output_dir.empty()) {
        if (!std::filesystem::exists(params.tensor_dbg_output_dir)) {
            if (!std::filesystem::create_directories(params.tensor_dbg_output_dir)) {
                LOG_ERR("%s: failed to create directory at %s\n",
                                __func__, params.tensor_dbg_output_dir.c_str());
                return EXIT_FAILURE;
            }
        }
    } else {
        LOG_WRN("%s: --output-dir parameter not set; observed tensors will not be saved!\n",
                       __func__);
    }

    run_debug(ctx, params.prompt);
    print_report(&cb_data);

    llama_backend_free();

    return EXIT_SUCCESS;
}
