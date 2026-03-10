#pragma once

// per-tensor info needed by the quantization work scheduler.
// constructed in llama-quant.cpp, passed to llama-quant-scheduler.cpp, not used otherwise.
struct tensor_sched_data {
    const std::vector<int64_t> ne;
    const ggml_type src_type;
    const ggml_type dst_type;
    const void * src_data; // pointer to raw source tensor data buffer, read-only
    const void * imatrix;  // pointer to imatrix data, or nullptr, read-only
    tensor_sched_data(
        const std::vector<int64_t> _ne,
        const ggml_type _src_type,
        const ggml_type _dst_type,
        const void * _src_data,
        const void * _imatrix
    )
    {
        
    }
};
