/**
 *
 * Capture intermediate tensors during graph execution and save them to disk in NumPy format.
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

int main(int argc, char ** argv) {
    GGML_UNUSED(argc); GGML_UNUSED(argv);
    common_init();

    llama_backend_init();

    // ...

    // llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
