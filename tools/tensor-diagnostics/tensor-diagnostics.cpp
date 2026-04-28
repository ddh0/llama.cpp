/**
 *
 * llama-tensor-diagnostics
 *
 * Capture all intermediate tensors during graph execution and save them to disk in NumPy format.
 *
 * This tool is intended primarily for debugging busted models, but may also be useful for analysis
 * of the hidden state `cur`, or any other node in the graph.
 *
 * The purpose of saving tensors in NumPy format rather than binary is to make debugging and
 * analysis easier - for example, we can capture multiple runs from different models and compare
 * them using a simple Python script. Once a tensor is saved to disk as a `.npy` file, then it
 * can be loaded directly as a NumPy array in any Python script using `np.load`.
 *
 * For more information, see the README in this directory, as well as the Python script.
 *
**/

#include "llama.h"
#include "common.h"
#include "arg.h"

#include <filesystem>

struct tensor_diagnostic_data {
    const size_t num_zeros;
    const size_t num_nans;
    const size_t num_infs; // XXX: NaN vs. inf ??
};

static bool tensor_diagnostic_cb(ggml_tensor * t, bool ask, void * user_data) {
    // TODO: cast ggml_tensor * t to const inside this function - we cannot modify it
    if (ask) {
        // the scheduler wants to know if we want to observe this tensor (we always do)
        return true;
    } else {
        // the scheduler is passing the tensor to us for observation
        const ggml_tensor & observed_tensor = *t;
        // if we return false, the scheduler will cancel the graph computation
    }
};

//
// main diagnostic operation:
//
//  1. read text from the file at the path specified by `text_input_file` as input to the model
//     (prompt processing)
//  2. autoregressively generate `n` new tokens after reading the input file
//     (text generation)
//  3. process every intermediate tensor (all nodes) with `tensor_diagnostic_cb` function, which
//     saves all output to the directory specified by `tensor_diagnostic_cb`
//
// results are saved under the directory specified by `diag_output_dir`, in subdirectories:
//  - `./pp/`: stores the tensors captured during prompt processing
//  - `./tg/`: stores the tensors captured during text generation
//
// if the directory specified by `diag_output_dir` already exists, the program will abort to prevent
// accidentally overwriting other diagnostic dumps.
//
static bool run_diagnostics(
    const std::filesystem::path text_input_file,
    const std::filesystem::path diag_output_dir,
    const int64_t max_n_pp,
    const int64_t max_n_tg)
{
    
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TENSOR_DIAGNOSTICS)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    tensor_diagnostic_data * cb_data;

    params.cb_eval = tensor_diagnostic_cb;
    params.cb_eval_user_data = &cb_data;

    auto common_init = common_init_from_params(params);
    auto llama_model = common_init->model();
    auto llama_ctx   = common_init->context();

    {
        res = run_diagnostics();
    }

    llama_perf_context_print(llama_ctx);
    llama_backend_free();

    return 0;
}
