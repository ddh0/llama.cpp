/**
 *
 * Whenever possible, we aim to overlap computation and tensor data disk I/O.
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

// pool of compute worker threads
struct compute_pool {
    const int32_t n_threads;
    std::vector<std::thread> threads;
    std::atomic_flag busy;

    compute_pool(const int32_t _n_threads):
        n_threads(_n_threads), threads(_n_threads) {
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
// goal: overlap I/O and computation as often as possible to speed-up the quantization process.
//
// the scheduler manages (`n_threads` + 2) threads:
// - 1 thread for the `read_worker`
// - `n_threads` threads for the `compute_pool` (tensor math is divided among compute workers)
// - 1 thread for the `write_worker`
//
struct scheduler {
    const int32_t n_threads;

    // per-tensor metadata for all tensors in the model
    std::vector<tensor_sched_data> tschd_vec;

    size_t largest_tensor_size_src = 0; // size of largest tensor to be quantized (as src type)
    size_t largest_tensor_size_f32 = 0; // size of largest tensor to be quantized (as f32)
    size_t largest_tensor_size_dst = 0; // size of largest tensor to be quantized (as dst type)

    //
    // scheduling pipeline buffers (one of each at most)
    //

    // size: largest_tensor_size_src
    std::vector<uint8_t> buf_read; // don't need this if using mmap?

    // size: largest_tensor_size_f32
    std::vector<float> buf_dequant; // dequantization buffer

    // size: largest_tensor_size_dst
    std::vector<uint8_t> buf_quant; // quantization buffer (do we really need this?)

    // size = largest tensor (as dst type)
    std::vector<uint8_t> buf_write; // hold tensor data for writing (NOTE: tensors must be in order in the output file)

    compute_pool pool;

    // initialize
    scheduler(const int32_t _n_threads, std::vector<tensor_sched_data> _tschd_vec):
        n_threads(_n_threads), tschd_vec(_tschd_vec), pool(_n_threads)
    {
        for (int32_t idx = 0; idx < tschd_vec.size(); idx++) {
        /*
            TODO: set these:
            largest_tensor_size         = ...;
            largest_tensor_size_dequant = ...;
            largest_tensor_size_quant   = ...;
        */
        }

        // TODO: allocate pipeline buffers
    };

    ~scheduler() {
        stop();
    }

    void start() {
        // TODO: start `read_worker` thread
        // TODO: THIS thread should manage the compute pool
        // TODO: start `write_worker` thread
        // return void when done, throw std::runtime_error if something fails
    }

    void stop() {
        // TODO: graceful shutdown + deallocation of buffers
    }

    void submit_compute(tensor_sched_data & tschd) {
        // TODO: 
    }
};
