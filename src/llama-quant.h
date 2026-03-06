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
    const ggml_type    src_type; // source tensor type
    const ggml_type    dst_type; // destination tensor type
    const int64_t      ne0;      // n_cols
    const int64_t      ne1;      // n_rows
    const int64_t      ne2;      // n_expert (or any 3rd tensor dimension)
    const int64_t      ne3;      // any 4th tensor dimension (currently unused, always 1)
    const void * const src_data; // pointer to raw source tensor data buffer, read-only
    const void * const imatrix;  // pointer to imatrix data, or nullptr, read-only
};
