#pragma once

// TODO: this can be optimzed out (currently used in both quantize.cpp and llama-quant.cpp)
struct tensor_quantization {
    std::string name;
    ggml_type quant = GGML_TYPE_COUNT;
};

// tensor categorization (used to avoid repeated string matching)
// NOTE: not all types of tensors are included here, this is fine
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

// current status of a tensor in the scheduler
enum tensor_quant_status {
    PENDING       = 0,  // waiting to be read into memory
    READY         = 1,  // loaded into memory, awaiting compute
    COMPUTING     = 2,  // currently computing
    PENDING_WRITE = 3,  // quantized, awating write (need to write in order)
    DONE          = 4   // written to the output stream
};

// per-tensor metadata. all members const except `status`, which is atomic.
struct tensor_metadata {
    const ggml_tensor const * t;    // const pointer to const tensor (src)
    const int32_t idx;              // index in sequence of all model tensors
    const tensor_category category; // tensor categorization, used to avoid repeated string matching
    const ggml_type src_type;       // source tensor type
    const ggml_type dst_type;       // destination tensor type
    const size_t src_size;          // source type tensor size, in bytes
    const size_t f32_size;          // f32 tensor size, in bytes
    const size_t dst_size;          // target type tensor size, in bytes
    const float * imatrix_data;     // importance matrix data, may be nullptr
    std::atomic<enum tensor_quant_status> status; // current status of this tensor in the scheduler
};
