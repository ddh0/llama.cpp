/**
 *
 * Whenever possible, we aim to overlap computation and tensor data disk I/O in the quantization
 * process.
 *
 * This is the primary bottleneck in very many cases, and currently it's not handled very
 * efficiently - computation never overlaps with I/O on `master` at the time of writing. Rather,
 * the code basically does:
 *
 *    load src tensor data -> dequantize and/or quantize -> write tensor data
 *
 * ...in a loop over all tensors. There is a great opportunity to improve it! I believe we may be
 * able to acheive a speedup of ~4x in _some_ cases by overlapping the work to be done. There are
 * many users quantizing many models with many billions of parameters - we don't want to leave any
 * performance on the table.
 *
**/

// #include "ggml-quants.h"
#include "llama.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-quant.h"

#include <stdint.h>
#include <stdexcept>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>

// pool of worker threads used for dequantization and quantization
struct compute_pool {
    const int32_t n_threads;
    std::vector<std::thread> threads;
    std::atomic_flag busy;

    compute_pool(const int32_t _n_threads):
        n_threads(_n_threads), threads(_n_threads)
    {
        // TODO: prepare the threads? but don't start them.
        // TODO: init `busy` atomic flag?
    };

    bool distribute(tensor_sched_data & data) {
        // TODO: distribute
    };
};

//
// quantization work scheduler
//
// goal: overlap I/O and computation as much as possible to speed up the quantization process.
//
// the scheduler manages (`n_threads` + 2) threads:
// - 1 thread for the `read_worker`
// - `n_threads` threads for the `compute_pool`
// - 1 thread for the `write_worker`
//
struct scheduler {
    const int32_t n_threads;

    // per-tensor metadata for all tensors in the model
    std::vector<tensor_sched_data> data_vec;

    size_t max_src_sz = 0; // size of largest tensor to be quantized (as src type)
    size_t max_f32_sz = 0; // size of largest tensor to be quantized (as float32)
    size_t max_dst_sz = 0; // size of largest tensor to be quantized (as dst type)

    //
    // scheduler pipeline buffers (one of each at most)
    //

    // size: max_src_sz
    std::vector<uint8_t> buf_read;    // don't need this if using mmap?
    // size: max_f32_sz
    std::vector<float>   buf_compute; // dequant/quant buffer
    // size = max_dst_sz
    std::vector<uint8_t> buf_write;   // hold tensor data for writing (NOTE: tensors must be in order in the output file)

    compute_pool pool;

    // init
    scheduler(const int32_t _n_threads, std::vector<tensor_sched_data> _data_vec):
        n_threads(_n_threads), data_vec(_data_vec), pool(_n_threads)
    {
        GGML_ASSERT(GGML_MAX_DIMS == 4 && "GGML_MAX_DIMS is not 4 - update this function");
        for (int32_t idx = 0; idx < data_vec.size(); ++idx) {
            const auto & data = data_vec[idx];
            const int64_t nrows = data.ne1 * data.ne2 * data.ne3;
            max_src_sz = std::max(max_src_sz, nrows * ggml_row_size(data.src_type, data.ne0));
            max_f32_sz = std::max(max_f32_sz, nrows * ggml_row_size(GGML_TYPE_F32, data.ne0));
            max_dst_sz = std::max(max_dst_sz, nrows * ggml_row_size(data.dst_type, data.ne0));
        }

        // TODO: allocate pipeline buffers
    };

    void start() {
        // TODO: start `read_worker` thread
        // TODO: THIS thread should manage the compute pool
        // TODO: start `write_worker` thread
        // throw std::runtime_error if something fails
    }

    void stop() {
        // TODO: graceful shutdown + deallocation of buffers
    }

    ~scheduler() {
        stop();
    }
};
