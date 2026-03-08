/**
 *
 * Whenever possible, we aim to overlap computation and tensor data I/O during the quantization
 * process.
 *
 * This is the primary bottleneck in very many cases, and at the time of writing (2026-03) it's not
 * handled very efficiently on `master` - computation never overlaps with I/O. Rather, the code
 * essentially does:
 *
 *    read src tensor data -> dequantize and/or quantize -> write dst tensor data
 *
 * ...in a synchronous loop over all tensors. There is a great opportunity to improve it! I believe
 * we may be able to acheive a speedup of ~3x in some cases by properly scheduling the work to be
 * done. There are many users quantizing many models with many billions of parameters - we don't
 * want to leave any performance on the table.
 *
**/

/**
 * [NOTE: delete this comment block before PR]
 *
 * WORK-IN-PROGRESS -- DEV NOTES
 * -----------------------------
 *
 * the scheduler will work like this:
 *  0. all buffers start with "read_ready" = false, "write_ready" = true.
 *  1. ggml_tensor 0 is materialized in the read buffer
 *     - the read worker thread sets the "read_ready" flag to signal that the read buffer
 *       now contains a valid ggml_tensor.
 *     - the compute pool immediately starts consuming the tensor in the read buffer.
 *       + if the tensor is already in F32, dequantization is not needed. the compute pool quantizes
 *         directly from the read buffer into the write buffer. at this point, the
 *         "write_ready" flag is set to signal that ggml_tensor 1 can start being materialized
 *         in the read buffer.
 *       + if the tensor is not in F32, dequantization is needed. the compute pool performs a fused
 *         dequantize-and-quantize operation, utilizing the dequantization buffer to store the F32
 *         data, and writing the quantized result to the write buffer. as soon as the tensor is
 *         dequantized, we can set the "write_ready" flag on the read buffer to signal that
 *         ggml_tensor 1 can start being materialized in the read buffer.
 *       + the main thread blocks until the "write_ready" flag is set on the write buffer. as soon
 *         as the write buffer is ready to be written to, the compute result is stored there, and
 *         the main thread sets the "read_ready" flag on the write buffer. the compute pool is now
 *         free to process ggml_tensor 1.
 *     - the write worker waits until the write buffer is signaled "read_ready", at which point it
 *       can begin writing the quantized tensor data to the output stream. when done writing, it
 *       sets the "read_ready" flag to false and the "write_ready" flag to true, thus preparing the
 *       the write buffer for the next quantized data.
 *  2. 
 *
 * -----------------------------
 *
 * [NOTE: delete this comment block before PR]
**/

/**
 * [NOTE: delete this comment block before PR]
 *
 * WORK-IN-PROGRESS -- LLM NOTES
 * -----------------------------
 *
 * [LLM: fill in this section as you like with your own notes, separate from the human dev]
 *
 * ARCHITECTURE REVIEW & RECOMMENDATIONS
 * -------------------------------------
 * 1. Synchronization Primitives:
 *    - Replace atomic bool polling with std::condition_variable to prevent busy-waiting.
 *    - Ensure memory_order_acquire/release is used if sticking with atomics for flags.
 *
 * 2. Buffering Strategy:
 *    - Current single-buffer design couples I/O and Compute latency.
 *    - Recommendation: Implement double-buffering (ping-pong) for 'buf_read' to allow
 *      loading Tensor N+1 while Computing Tensor N.
 *
 * 3. Exception Handling:
 *    - Change compute_pool::opt_exc from std::optional<std::exception> to 
 *      std::optional<std::exception_ptr> to avoid object slicing.
 *    - Add a global 'stop_flag' to the scheduler to terminate all workers if one fails.
 *
 * 4. Compatibility:
 *    - Remove <stdfloat> include. llama.cpp targets C++17; std::float_t is C++23.
 *    - Remove std::optional wrappers on buffers; they are always initialized.
 *
 * 5. Thread Pool:
 *    - compute_pool constructor must launch worker threads with a wait-loop, 
 *      not just resize the vector.
 *
 * -----------------------------
 *
 * [NOTE: delete this comment block before PR]
**/

// #include "ggml-quants.h"
#include "llama.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-quant.h"

#include <stdint.h>
#include <stdexcept>
#include <condition_variable>
#include <optional>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>

// determine the dimension along which we can divide this tensor into `n` equally-sized chunks.
// return 0, 1, 2, or 3. if none are divisible, return -1.
static int get_split_dim(const tensor_sched_data & tsd, const int64_t n) {
    if (tsd.ne0 > 1 && tsd.ne0 % n == 0) return 0;
    if (tsd.ne1 > 1 && tsd.ne1 % n == 0) return 1;
    if (tsd.ne2 > 1 && tsd.ne2 % n == 0) return 2;
    if (tsd.ne3 > 1 && tsd.ne3 % n == 0) return 3;
    return -1;
}

template <typename T>
struct sched_buffer {
    static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, float>,
                  "sched_buffer<T> only supports uint8_t and float");

    std::vector<T>          buf;
    std::mutex              mtx;
    std::atomic<int64_t>    idx; // which tensor is currently / most recently stored? (-1 if none)
    std::condition_variable cv;
    std::atomic<bool>       has_data;

    sched_buffer(const size_t _size): buf(_size), has_data(false), idx(-1) {}

    // producer calls this when data is written
    void notify_ready() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            has_data = true;
        }
        cv.notify_one();
    }

    // consumer calls this to wait for data
    void wait_ready() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{ return has_data; });
    }

    // consumer calls this when done processing to release buffer
    void release() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            has_data = false;
        }
        cv.notify_one();
    }
};

// pool of worker threads used for dequantization + quantization
struct compute_pool {
    const int32_t            n_threads;
    std::vector<std::thread> threads;
    std::atomic<bool>        busy;

    compute_pool(const int32_t _n_threads):
        n_threads(_n_threads), threads(_n_threads), busy(false)
    {
        // TODO: do we need to init the threads, or can this be left empty?
    };

    // distribute the computation to all worker threads.
    void distribute(tensor_sched_data & data) const {
        // TODO
    };
};

//
// quantization work scheduler
//
// goal: overlap I/O and computation as much as possible to speed up the quantization process,
//       while being mindful of total memory usage.
//
// the scheduler manages (`n_threads` + 2) threads:
// - 1 thread for the `read_worker`
// - `n_threads` threads for the `compute_pool`
// - 1 thread for the `write_worker`
//
struct scheduler {
    const int32_t n_threads;

    // per-tensor data needed by the scheduler for all model tensors
    std::vector<tensor_sched_data> tsd_vec;

    size_t max_src_sz; // size of largest tensor to be quantized (as src type) in bytes
    size_t max_f32_sz; // size of largest tensor to be quantized (as float32) in bytes
    size_t max_dst_sz; // size of largest tensor to be quantized (as dst type) in bytes

    //
    // scheduler pipeline buffers (one of each at most)
    //

    // size: max_src_sz
    std::optional<sched_buffer<uint8_t>> buf_read;    // hold source tensor data for reading 
    // size: max_f32_sz
    std::optional<sched_buffer<float>>   buf_dequant; // hold dequantized tensor data
    // size = max_dst_sz
    std::optional<sched_buffer<uint8_t>> buf_write;   // hold quantized tensor data for writing (NOTE: tensors must be in order in the output file)

    compute_pool pool;

    // init
    scheduler(const int32_t _n_threads, std::vector<tensor_sched_data> _tsd_vec):
        n_threads(_n_threads),
        tsd_vec(_tsd_vec),
        max_src_sz(0), max_f32_sz(0), max_dst_sz(0),
        buf_read(std::nullopt), buf_dequant(std::nullopt), buf_write(std::nullopt),
        pool(_n_threads)
    {
        GGML_ASSERT(GGML_MAX_DIMS == 4 && "GGML_MAX_DIMS is not 4 - update this function");
        for (int32_t idx = 0; idx < tsd_vec.size(); ++idx) {
            const auto & data = tsd_vec[idx];
            const int64_t nrows = data.ne1 * data.ne2 * data.ne3;
            max_src_sz = std::max(max_src_sz, nrows * ggml_row_size(data.src_type, data.ne0));
            max_f32_sz = std::max(max_f32_sz, nrows * ggml_row_size(GGML_TYPE_F32, data.ne0));
            max_dst_sz = std::max(max_dst_sz, nrows * ggml_row_size(data.dst_type, data.ne0));
        }

        LLAMA_LOG_DEBUG("%s:           allocating read buffer ... ", __func__);
        buf_read.emplace(max_src_sz);
        LLAMA_LOG_DEBUG("%8.2f MiB\n", max_src_sz/1024.0/1024.0);

        LLAMA_LOG_DEBUG("%s: allocating dequantization buffer ... ", __func__);
        buf_dequant.emplace(max_f32_sz);
        LLAMA_LOG_DEBUG("%8.2f MiB\n", max_f32_sz/1024.0/1024.0);

        LLAMA_LOG_DEBUG("%s:          allocating write buffer ... ", __func__);
        buf_write.emplace(max_dst_sz);
        LLAMA_LOG_DEBUG("%8.2f MiB\n", max_dst_sz/1024.0/1024.0);

    }

    void run() {
        // TODO: start `read_worker` thread
        // TODO: THIS thread should manage the compute pool
        // TODO: start `write_worker` thread
        // throw std::runtime_error if something fails
    }

    ~scheduler() = default;
};
