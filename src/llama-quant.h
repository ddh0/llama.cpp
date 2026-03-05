#pragma once
// #include <string>
// #include <stdint.h>
// #include "ggml.h"

// store result of parsing --tensor-type option
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

// per-tensor info needed by the quantization work scheduler for efficient quantization.
// constructed in llama-quant.cpp, passed to llama-quant-scheduler.cpp, not used otherwise.
struct tensor_sched_data {
    const int64_t ne0; // ncols
    const int64_t ne1; // nrows
    const int64_t ne2; // n_expert (or any other 3rd dimension)
    const int64_t ne3; // 4D (currently unused)
    const ggml_type src_type;
    const ggml_type dst_type;
};
