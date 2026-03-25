#pragma once

enum sched_cmd_status {
    CMD_STATUS_PENDING,
    CMD_STATUS_IN_PROGRESS,
    CMD_STATUS_COMPLETE,
    CMD_STATUS_COUNT, // always last
};

// types of operations that performed the quantization work scheduler
enum sched_cmd_type {
    CMD_TYPE_READ,
    CMD_TYPE_DEQUANTIZE,
    CMD_TYPE_QUANTIZE,
    CMD_TYPE_WRITE,
    CMD_TYPE_COUNT // always last
};

// unit of work for the quantization work scheduler (command pattern)
struct sched_cmd {
    const ggml_tensor * tensor;
    const enum sched_cmd_type sched_cmd_type;

    std::atomic<enum sched_cmd_status> sched_cmd_status;
};
