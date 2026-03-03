/* llama-quant-scheduler.cpp -- C++17

ASPIRATIONS:

Whenever possible, we must overlap computation and disk I/O. In fact, disk I/O is the main
bottleneck in almost all cases, and currently on `master` it's not handled very well - computation
never overlaps with I/O. So there is some great opportunity to improve it. Though I'm really not
sure what the cleanest, simplest approach would be here, in combination with the scheduler. Should
I/O _also_ be coordinated by the scheduler? Would it be feasible to have the main thread handle the
I/O queue as completed by the scheduler? I think so. I hope so.

At the time of writing (2026-03-02), in the code on `master`, there is no notion of overlapping
computation and I/O. It is kept simple (if a bit messy) and it just does (load src data -> (convert
to f32) -> quantize to target type -> write tensor data) in a for loop over all tensors. I believe
we may be able to acheive a speedup of ~4x in _some_ cases by managing the work to be done more
effectively. There are many people quantizing many models every day with untold billions of
parameters - we don't want to leave any wall-clock time on the table.

NOTE: The tensors must end up in order in the output GGUF.
*/

#include "ggml-quants.h"
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

struct thread_pool {
    const int32_t            n_workers;
    std::vector<std::thread> workers;
    std::atomic<bool> busy = false;

    thread_pool(const int32_t _n_workers):
        n_workers(_n_workers) {
        // TODO: prepare the workers but don't start them
    };

    void start() {
        // TODO: start the workers
    };

    void stop() {
        // TODO
    };
};

//
// quantization work scheduler
//
// goal: overlap I/O and computation, keep all threads busy (as much as reasonably possible)
//
// the scheduler manages (`n_threads` + 2) threads
// - 1 thread for the `read_worker`
// - `n_thread` threads for the `thread_pool` (tensor math is divided among compute workers)
// - 1 thread for the `write_worker`
//
struct scheduler {
    const int32_t n_workers;

    // metadata for all tensors in the model
    std::vector<tensor_metadata> tm_vec;

    //
    // scheduling pipeline buffers (1 of each at most)
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

    thread_pool pool;

    // initialize
    scheduler(const int32_t _n_workers, std::vector<tensor_metadata> _tm_vec):
        n_workers(_n_workers), tm_vec(_tm_vec), pool(_n_workers)
    {
        for (int32_t idx; idx < tm_vec.size(); idx++) {
            if (tm_vec[idx].src_size > largest_tensor_size) {
                largest_tensor_size = tm_vec[idx].src_size;
            }

            const auto ne = tm_vec[idx].t->ne[0] *
                            tm_vec[idx].t->ne[1] *
                            tm_vec[idx].t->ne[2] *
                            tm_vec[idx].t->ne[3];

            if (ne > largest_tensor_ne) {
                largest_tensor_ne = ne;
            }
        }

        // buf_dequant.reserve()
    };

    void run() {
        // TODO: start thread to handle the read_queue
        // TODO: THIS thread should manage the compute worker pool
        // TODO: start thread to handle the write_queue
        // return void when done, throw std::runtime_error if something fails
    }
};
