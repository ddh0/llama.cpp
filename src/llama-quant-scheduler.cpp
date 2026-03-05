/* llama-quant-scheduler.cpp -- C++17

ASPIRATIONS
-----------

Whenever possible, we must overlap computation and disk I/O. In fact, disk I/O is the main
bottleneck in very many cases, and currently on `master` it's not handled very well - computation
never overlaps with I/O. There is a great opportunity to improve it!

At the time of writing (2026-03-02), the code on `master` is kept simple (if a bit messy) and it
simply does...

    load src data -> (convert to f32) -> quantize to target type -> write tensor data

...in a for loop over all tensors. I believe we may be able to acheive a speedup of ~4x in _some_
cases by managing the work to be done more effectively. There are many people quantizing many models
every day with untold billions of parameters - we don't want to leave any performance on the table.

The quantized tensors MUST end up in order in the output GGUF.
*/

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

    void start() {
        // TODO: start the threads (but wait for work, don't spin)
    };

    void stop() {
        // TODO: forcibly stop the thread pool (called should check `busy` before doing this)
    };
};

//
// quantization work scheduler
//
// goal: overlap I/O and computation, keep all threads busy (as much as reasonably possible)
//
// the scheduler actually manages (`n_threads` + 2) threads:
// - 1 thread for the `read_worker`
// - `n_thread` threads for the `thread_pool` (tensor math is divided among compute workers)
// - 1 thread for the `write_worker`
//
struct scheduler {
    const int32_t n_threads;

    // metadata for all tensors in the model
    std::vector<tensor_sched_data> tschd_vec;

    //
    // scheduling pipeline buffers (one of each at most)
    //

    // don't need this if using mmap
    std::vector<uint8_t> buf_read; // size = largest tensor (as found) (`largest_tensor_size`)

    // dequantization compute buffer
    std::vector<float> buf_dequant; // size = largest tensor (as f32) (`largest_tensor_size_dequant`)

    // quantization compute buffer
    std::vector<uint8_t> buf_quant; // size = largest tensor (quantized) (`largest_tensor_size_quant`)

    // hold tensor data (NOTE: tensors must be in order in the output file)
    std::vector<uint8_t> buf_write; // size = largest tensor (quantized)

    size_t largest_tensor_size         = 0;
    size_t largest_tensor_size_dequant = 0;
    size_t largest_tensor_size_quant   = 0;

    compute_pool pool;

    // initialize
    scheduler(const int32_t _n_threads, std::vector<tensor_sched_data> _tschd_vec):
        n_threads(_n_threads), tschd_vec(_tschd_vec), pool(_n_threads)
    {
        for (int32_t idx; idx < tschd_vec.size(); idx++) {
        /*
            TODO: set these:
            largest_tensor_size         = ...;
            largest_tensor_size_dequant = ...;
            largest_tensor_size_quant   = ...;
        */
        }

        // TODO: reserve pipeline buffers
    };

    void run() {
        // TODO: start `read_worker` thread
        // TODO: THIS thread should manage the compute pool
        // TODO: start `write_worker` thread
        // return void when done, throw std::runtime_error if something fails
    }
};
