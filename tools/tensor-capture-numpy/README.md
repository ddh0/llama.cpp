# llama-tensor-capture-numpy

Capture intermediate tensors during graph execution and save them to disk in NumPy format.

This tool is intended primarily for debugging busted models, but may also be useful for analysis of the hidden state `cur`, or any other node in the graph.

The purpose of saving tensors in NumPy format rather than binary is to make debugging and analysis easier - for example, we can capture multiple runs from different models and compare them using a simple Python script. Once a tensor is saved to disk as a `.npy` file, then it can be directly loaded as a NumPy array in any Python script.

<!-- TODO -->
