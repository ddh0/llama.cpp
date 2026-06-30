// ================================================================================================
// playground.cpp
// ================================================================================================
//
// this file is not meant to be incorporated into llama.cpp!
//
// it is only meant as a development and testing sandbox.
//
// ================================================================================================

#include "llama.h"
#include "log.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "common.h"
#include "gguf.h"

#include <stdint.h>
#include <random>
#include <cmath>

#define TDIM "%7" PRId64 // tensor dimension format string
#define TNAME "%-54s"    // tensor name format string

// elements with absolute values strictly smaller than this are considered to be zero
static constexpr float ZERO_TOLERANCE = 0.005;

static std::mt19937_64 & global_rng() {
    static std::random_device rd;
    static std::mt19937_64 rng(rd());
    return rng;
}

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
    switch (t->type) {
        case GGML_TYPE_F32: {
            auto f32_data = static_cast<const float *>(t->data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_flt(f32_data[i]);
            }
        } break;
        case GGML_TYPE_F16: {
            auto f16_data = static_cast<const ggml_fp16_t *>(t->data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_flt(ggml_fp16_to_fp32(f16_data[i]));
            }
        } break;
        case GGML_TYPE_BF16: {
            auto bf16_data = static_cast<const ggml_bf16_t *>(t->data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_flt(ggml_bf16_to_fp32(bf16_data[i]));
            }
        } break;
        case GGML_TYPE_I64: {
            auto i64_data = static_cast<const int64_t *>(t->data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_int(i64_data[i]);
            }
        } break;
        case GGML_TYPE_I32: {
            auto i32_data = static_cast<const int32_t *>(t->data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_int(i32_data[i]);
            }
        } break;
        case GGML_TYPE_I16: {
            auto i16_data = static_cast<const int16_t *>(t->data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_int(i16_data[i]);
            }
        } break;
        case GGML_TYPE_I8: {
            auto i8_data = static_cast<const int8_t *>(t->data);
            for (size_t i = 0; i < n_elements; ++i) {
                stats.read_int(i8_data[i]);
            }
        } break;
        default:
            LOG_WRN("%s: tensor '" TNAME "' has unsupported type (%s), stats will be empty\n",
                    __func__, t->name, ggml_type_name(t->type));
            return stats;
    }

    GGML_ASSERT(stats.n_elem == n_elements); // sanity check
    return stats;
}

#include <sstream>
#include <iomanip>
#include <cstdint>

std::string tensor_get_addr(const ggml_tensor * t) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setfill('0') << std::setw(sizeof(void *) * 2) << static_cast<const void *>(t);
    return oss.str();
}

void print_tensor(const ggml_tensor * t, bool include_stats) {
    GGML_ASSERT(t != nullptr);
    GGML_ASSERT(!ggml_is_empty(t) && "tensor is empty");
    const auto t_stats = get_tensor_stats(t);
    const bool t_all_zero = t_stats.n_zeros == t_stats.n_elem;
    if (t_all_zero || t_stats.n_infs > 0 || t_stats.n_nans > 0) {
        LOG_WRN(
            "%4s: " TNAME " [ " TDIM ", " TDIM ", " TDIM ", " TDIM " ] all zero = %1s, "
            "infs = %7zu, nans = %5zu, min = %9.2f, avg = %6.2f, max = %9.2f, var = %9.2f\n",
            ggml_type_name(t->type), t->name, t->ne[0], t->ne[1], t->ne[2],
            t->ne[3], t_all_zero ? "1" : "0", t_stats.n_infs, t_stats.n_nans, t_stats.min_val,
            t_stats.mean, t_stats.max_val, t_stats.get_variance());
    } else {
        LOG_INF(
            "%4s: " TNAME " [ " TDIM ", " TDIM ", " TDIM ", " TDIM " ] all zero = %1s, "
            "infs = %7zu, nans = %5zu, min = %9.2f, avg = %6.2f, max = %9.2f, var = %9.2f\n",
            ggml_type_name(t->type), t->name, t->ne[0], t->ne[1], t->ne[2],
            t->ne[3], t_all_zero ? "1" : "0", t_stats.n_infs, t_stats.n_nans, t_stats.min_val,
            t_stats.mean, t_stats.max_val, t_stats.get_variance());
    }
}

// helper function for lazy developers
ggml_tensor * new_tensor(struct ggml_context * ctx, const int64_t ncols, const int64_t nrows, const int64_t ne2, const int64_t ne3) {
    return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, /* ne0 */ ncols, /* ne1 */ nrows, ne2, ne3);
}

// overwrite one column of a 2D tensor with values drawn from a gaussian distribution
void column_fill_gaussian(ggml_tensor * t, const int64_t col_idx, float mean, float stddev) {
    GGML_ASSERT(t != nullptr);
    GGML_ASSERT(!ggml_is_empty(t) && "tensor is empty");
    GGML_ASSERT(t->type == GGML_TYPE_F32 && "only F32 tensors are supported");
    GGML_ASSERT(t->ne[2] == t->ne[3] == 1 && "only 2D tensors are supported");
    GGML_ASSERT(col_idx >= 0 && col_idx < t->ne[0] && "invalid column index");

    std::normal_distribution<float> dist(mean, stddev);

    const int64_t ncols = t->ne[0];
    const int64_t nrows = t->ne[1];

    auto data = static_cast<float *>(t->data);

    for (int64_t i1 = 0; i1 < nrows; i1++) {
        data[col_idx + i1 * ncols] = dist(global_rng());
    }
}

// overwrite one column of a 2D tensor with values drawn from a uniform distribution
void column_fill_uniform(ggml_tensor * t, const int64_t col_idx, float min, float max) {
    GGML_ASSERT(t != nullptr);
    GGML_ASSERT(!ggml_is_empty(t) && "tensor is empty");
    GGML_ASSERT(t->type == GGML_TYPE_F32 && "only F32 tensors are supported");
    GGML_ASSERT(t->ne[2] == t->ne[3] == 1 && "only 2D tensors are supported");
    GGML_ASSERT(col_idx >= 0 && col_idx < t->ne[0] && "invalid column index");

    const int64_t ncols = t->ne[0];
    const int64_t nrows = t->ne[1];

    std::uniform_real_distribution<float> dist(min, max);

    auto data = static_cast<float *>(t->data);

    for (int64_t i1 = 0; i1 < nrows; i1++) {
        data[col_idx + i1 * ncols] = dist(global_rng());
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C"); // ref: https://github.com/ggml-org/llama.cpp/pull/17331
}
