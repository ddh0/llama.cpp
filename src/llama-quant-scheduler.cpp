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
 * -----------------------------
 *
 * [NOTE: delete this comment block before PR]
**/

#include "llama.h"
#include "llama-impl.h"
#include "llama-quant.h"

#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <stdint.h>
#include <type_traits>
#include <condition_variable>

// determine the dimension along which we can divide this tensor into `n` equally-sized chunks.
// return 0, 1, 2, or 3. if none are divisible, return -1.
static int get_split_dim(const tensor_sched_data & tsd, const int64_t n) {
    if (tsd.ne0 > n && tsd.ne0 % n == 0) return 0;
    if (tsd.ne1 > n && tsd.ne1 % n == 0) return 1;
    if (tsd.ne2 > n && tsd.ne2 % n == 0) return 2;
    if (tsd.ne3 > n && tsd.ne3 % n == 0) return 3;
    return -1;
}

template <typename T> struct sched_buffer {
    static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, float>,
                  "sched_buffer<T> only supports uint8_t and float");

    std::vector<T>          buf;
    std::mutex              mtx;
    std::atomic<int64_t>    idx; // which tensor is currently or most recently stored? (-1 at init, then 0 for 1st tensor, 1 for 2nd tensor...)
    std::atomic<bool>       has_data;
    std::condition_variable cv;

    // init but don't allocate the buffer yet
    sched_buffer():
        has_data(false), idx(-1)
    {}

    // allocate the buffer and return the allocated size in bytes
    size_t allocate(const size_t _size) {
        buf.resize(_size);
        return sizeof(T) * _size;
    }

    // signal to workers that this buffer now has data for tensor at index `_idx`.
    // this updates the buffer's `idx` to match. all indices must be sequential.
    void signal_has_data(const int64_t _idx) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            GGML_ASSERT(_idx == idx + 1 && "buffer tensor indices must be sequential");
            has_data = true;
            idx = _idx;
        }
        cv.notify_one();
    }

    // workers call this function to wait for data in this buffer.
    void wait_for_data() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]{ return has_data; });
    }

    // signal to workers that this buffer should no longer be read from.
    void signal_no_data() {
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

    size_t max_src_sz = 0; // size of largest tensor to be quantized (as src type) in bytes
    size_t max_f32_sz = 0; // size of largest tensor to be quantized (as float32) in bytes
    size_t max_dst_sz = 0; // size of largest tensor to be quantized (as dst type) in bytes

    //
    // scheduler pipeline buffers (one of each at most)
    //

    // size: max_src_sz
    sched_buffer<uint8_t> buf_read;        // tensor data is read into here as fast as possible (read worker keeps it full).
    // size: max_src_sz
    sched_buffer<uint8_t> buf_compute_src; // compute workers read src tensor data from here
    // size: max_f32_sz
    sched_buffer<float>   buf_compute_f32; // intermediate f32 tensor data (if necessary)
    // size = max_dst_sz
    sched_buffer<uint8_t> buf_compute_dst; // compute workers write dst tensor data into here
    // size = max_dst_sz
    sched_buffer<uint8_t> buf_write;       // tensor data is written to the output stream IN ORDER by the write worker.

    compute_pool pool;

    // init
    scheduler(const int32_t _n_threads, std::vector<tensor_sched_data> _tsd_vec):
        n_threads(_n_threads),
        tsd_vec(_tsd_vec),
        pool(_n_threads)
    {
        GGML_ASSERT(GGML_MAX_DIMS == 4 && "GGML_MAX_DIMS is not 4 - update this function");
        for (int32_t idx = 0; idx < tsd_vec.size(); ++idx) {
            const auto & data = tsd_vec[idx];
            const size_t nrows = data.ne1 * data.ne2 * data.ne3;
            max_src_sz = std::max(max_src_sz, nrows * ggml_row_size(data.src_type, data.ne0));
            max_f32_sz = std::max(max_f32_sz, nrows * ggml_row_size(GGML_TYPE_F32, data.ne0));
            max_dst_sz = std::max(max_dst_sz, nrows * ggml_row_size(data.dst_type, data.ne0));
        }

        LLAMA_LOG_DEBUG("%s:        allocating read buffer ... ", __func__);
        GGML_ASSERT(max_src_sz == buf_read.allocate(max_src_sz)); 
        LLAMA_LOG_DEBUG("%8.2f MiB\n", max_src_sz/1024.0/1024.0);

        LLAMA_LOG_DEBUG("%s: allocating compute src buffer ... ", __func__);
        GGML_ASSERT(max_src_sz == buf_compute_src.allocate(max_src_sz)); 
        LLAMA_LOG_DEBUG("%8.2f MiB\n", max_src_sz/1024.0/1024.0);

        LLAMA_LOG_DEBUG("%s: allocating compute f32 buffer ... ", __func__);
        GGML_ASSERT(max_f32_sz == buf_compute_f32.allocate(max_f32_sz / sizeof(float)));
        LLAMA_LOG_DEBUG("%8.2f MiB\n", max_f32_sz/1024.0/1024.0);

        LLAMA_LOG_DEBUG("%s: allocating compute dst buffer ... ", __func__);
        GGML_ASSERT(max_dst_sz == buf_compute_dst.allocate(max_dst_sz));
        LLAMA_LOG_DEBUG("%8.2f MiB\n", max_dst_sz/1024.0/1024.0);

        LLAMA_LOG_DEBUG("%s:       allocating write buffer ... ", __func__);
        GGML_ASSERT(max_dst_sz == buf_write.allocate(max_dst_sz));
        LLAMA_LOG_DEBUG("%8.2f MiB\n", max_dst_sz/1024.0/1024.0);
    }

    void run() {
        // TODO: start `read_worker` thread
        // TODO: start `compute_worker` thread (?)
        // TODO: start `write_worker` thread
        // throw std::runtime_error if something fails
    }

    ~scheduler() = default;
};
