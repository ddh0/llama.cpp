#pragma once
#include <string>

// result of parsing --tensor-type option
struct tensor_type_option {
    std::string name;
    ggml_type   type = GGML_TYPE_COUNT;
};

// tensor categorization - used to avoid repeated string matching in quantization logic.
// this is different from LLM_TN - we want broad categories, not specific tensor names per arch.
enum class tensor_category {
    TOKEN_EMBD,
    ATTENTION_Q,
    ATTENTION_V,
    ATTENTION_K,
    ATTENTION_QKV,
    ATTENTION_KV_B,
    ATTENTION_OUTPUT,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    OUTPUT,
    OTHER
};

// per-tensor info needed by the quantization work scheduler.
// constructed in llama-quant.cpp, passed to llama-quant-scheduler.cpp, not used otherwise.
struct tensor_sched_data {
    const void * const src_data; // pointer to raw source tensor data, read-only
    const ggml_type    src_type;
    const ggml_type    dst_type;
    const int64_t      ne0; // ncols
    const int64_t      ne1; // nrows
    const int64_t      ne2; // n_expert (or any 3rd tensor dimension)
    const int64_t      ne3; // any 4th tensor dimension (currently unused, always 1)
};
