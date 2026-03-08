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

// #include "ggml-quants.h"
#include "llama.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-quant.h"

#include <stdint.h>
#include <stdfloat>
#include <stdexcept>
#include <optional>
#include <thread>
#include <array>
#include <vector>
#include <atomic>
#include <mutex>

// return the dimension along which we can divide this tensor into `n` equally-sized chunks.
// return -1 if none are divisible.
static int get_split_dimension(const tensor_sched_data & tsd, const int64_t n) {
    if (tsd.ne0 > 1 && tsd.ne0 % n == 0) return 0;
    if (tsd.ne1 > 1 && tsd.ne1 % n == 0) return 1;
    if (tsd.ne2 > 1 && tsd.ne2 % n == 0) return 2;
    if (tsd.ne3 > 1 && tsd.ne3 % n == 0) return 3;
    return -1;
}

template <typename T>
struct sched_buffer {
    size_t size;
    std::vector<T> buf;
    std::atomic<bool> write_ready;
    std::atomic<bool> read_ready;
    std::atomic<int64_t> idx;

    sched_buffer() : size(0), buf(), write_ready(true), read_ready(false), idx(-1) {}

    void init(const size_t _size) {
        size = _size;
        buf = std::vector<T>(_size);
        write_ready = true;
        read_ready = false;
        idx = -1;
    }

    void reset() {
        buf.clear();
        write_ready = true;
        read_ready = false;
        idx = -1;
    };

    ~sched_buffer() = default;
};

// pool of worker threads used for dequantization + quantization
struct compute_pool {
    const int32_t n_threads;
    std::vector<std::thread> threads;
    std::atomic<bool> busy;
    std::optional<std::exception> opt_exc;

    compute_pool(const int32_t _n_threads):
        n_threads(_n_threads), threads(_n_threads)
    {};

    // distribute the computation to all worker threads.
    // return an exception, if one occured during computation, nullopt otherwise.
    std::optional<std::exception> distribute(tensor_sched_data & data) {
        // TODO
    };
};

//
// quantization work scheduler
//
// goal: overlap I/O and computation as much as possible to speed up the quantization process,
//       while still being mindful of total memory usage.
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

    size_t max_src_sz = 0; // size of largest tensor to be quantized (as src type) in bytes
    size_t max_f32_sz = 0; // size of largest tensor to be quantized (as float32) in bytes
    size_t max_dst_sz = 0; // size of largest tensor to be quantized (as dst type) in bytes

    //
    // scheduler pipeline buffers (one of each at most)
    //

    // size: max_src_sz
    sched_buffer<uint8_t> buf_read;    // hold source tensor data for reading 
    // size: max_f32_sz
    sched_buffer<float>   buf_dequant; // hold dequantized tensor data
    // size = max_dst_sz
    sched_buffer<uint8_t> buf_write;   // hold quantized tensor data for writing (NOTE: tensors must be in order in the output file)

    compute_pool pool;

    // init
    scheduler(const int32_t _n_threads, std::vector<tensor_sched_data> _data_vec):
        n_threads(_n_threads),
        data_vec(_data_vec),
        pool(_n_threads)
    {
        GGML_ASSERT(GGML_MAX_DIMS == 4 && "GGML_MAX_DIMS is not 4 - update this function");
        for (int32_t idx = 0; idx < data_vec.size(); ++idx) {
            const auto & data = data_vec[idx];
            const int64_t nrows = data.ne1 * data.ne2 * data.ne3;
            max_src_sz = std::max(max_src_sz, nrows * ggml_row_size(data.src_type, data.ne0));
            max_f32_sz = std::max(max_f32_sz, nrows * ggml_row_size(GGML_TYPE_F32, data.ne0));
            max_dst_sz = std::max(max_dst_sz, nrows * ggml_row_size(data.dst_type, data.ne0));
        }

        LLAMA_LOG_DEBUG("%s:           allocating read buffer ... ", __func__);
        buf_read.init(max_src_sz);
        LLAMA_LOG_DEBUG("%8.2f MiB\n", max_src_sz/1024.0/1024.0);

        LLAMA_LOG_DEBUG("%s: allocating dequantization buffer ... ", __func__);
        buf_dequant.init(max_f32_sz);
        LLAMA_LOG_DEBUG("%8.2f MiB\n", max_f32_sz/1024.0/1024.0);

        LLAMA_LOG_DEBUG("%s:          allocating write buffer ... ", __func__);
        buf_write.init(max_dst_sz);
        LLAMA_LOG_DEBUG("%8.2f MiB\n", max_dst_sz/1024.0/1024.0);

    };

    void start() {
        // TODO: start `read_worker` thread
        // TODO: THIS thread should manage the compute pool
        // TODO: start `write_worker` thread
        // throw std::runtime_error if something fails
    }

    void stop() {
        LLAMA_LOG_DEBUG("%s: deallocating buffers ... ", __func__);
        LLAMA_LOG_DEBUG("done\n");
    }

    ~scheduler() {
        stop();
    }
};
