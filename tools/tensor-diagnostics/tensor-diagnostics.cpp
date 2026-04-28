/**
 *
 * llama-tensor-diagnostics
 *
 * This tool observes ALL intermediate tensors during graph execution, for a configurable number of
 * prompt processing and text generation batches. It will display info about...
 *
 *   ... tensors with ANY non-finite elements
 *
 *   ... tensors with ANY NaN elements
 *
 *   ... tensors whose elements are ALL zero (within tolerance [-0.005, 0.005] inclusive)
 *
 * All observed tensors will be captured, converted to the NumPy v1.0 format, and saved to disk.
 *
 * The resulting `.npy` tensor files can be loaded directly as a `np.ndarray` in Python using the 
 * `np.load` function. From this point, in-depth post-mortem analysis of tensor data is hopefully
 * much easier.
 *
**/

#include "llama-impl.h" // needed for LLAMA_LOG_INFO etc.
#include "llama.h"
#include "common.h"
#include "arg.h"

#include <filesystem>

struct tensor_diagnostic_data {
    size_t num_nans = 0;
    size_t num_infs = 0;
    bool   all_zero = true;
};

// callback function receives tensors from scheduler
static bool tensor_diagnostic_cb(ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        //
        // before graph compute, the scheduler asks us if we want to observe each tensor.
        // since this tool is primarily intended for diagnosing buggy or broken models (and not for
        // general-purpose use by end-users), it is probably preferable to ALWAYS observe ALL
        // tensors that we possibly can.
        //
        // in the future, we can almost certainly safely ignore certain types of tensors,
        // particularly permutations and reshapings of tensors that we _did not_ ignore. currently,
        // while this tool is under development, we will signal that we want to observe all tensors.
        //
        return true;
    } else {
        // we are in-flight; the scheduler is passing us the real tensor for observation.
        //
        // TODO: do stuff
        //
        return true; // if we return false, the scheduler aborts graph computation
    }
};

// process a single tensor and return diagnostic data
static tensor_diagnostic_data process_tensor(const ggml_tensor * t) {
    // TODO
}

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
    llama_context * ctx,
    const std::filesystem::path text_input_file,
    const std::filesystem::path diag_output_dir,
    const int32_t n_pp_batches,
    const int32_t n_tg_batches
) {
    LLAMA_LOG_INFO("running diagnostics ... this may take a while ...\n");

    LLAMA_LOG_INFO("running diagnostics ... this may take a while ...\n");
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

    common_init_result_ptr common_init = common_init_from_params(params);

    llama_model * model = common_init->model(); GGML_ASSERT(model != nullptr);
    llama_context * ctx = common_init->context(); GGML_ASSERT(ctx != nullptr);

    bool success;

    {
        success = run_diagnostics(
            /* llama_context = */ ctx,
            /* prompt_file   = */ std::filesystem::path(params.prompt_file), // --file or -f
            /* output_dir    = */ std::filesystem::path("./diag-output/"), // TODO: make configurable
            /* n_pp_batches  = */ int32_t(1), // TODO: make configurable
            /* n_tg_batches  = */ int32_t(1)  // TODO: make configurable
        );
    }

    llama_perf_context_print(ctx);
    llama_backend_free();

    return (int)success;
}
