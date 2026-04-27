// Source for the graph_do_while condition kernel.
//
// After editing, regenerate the pre-built fatbin:
//
//   python scripts/build_condition_kernel_fatbin.py

#include <cstdint>
#include <cuda_runtime.h>

// Condition kernel for graph_do_while conditional while nodes.
//
// Reads the user's i32 loop-control flag from GPU memory via an indirection
// slot, and tells the CUDA graph's conditional while node whether to run
// another iteration.
//
// The indirection allows swapping the counter ndarray between graph launches
// without rebuilding: the slot's device address is baked into the graph, but
// the pointer it contains can be updated via memcpy before each launch.
//
// Parameters:
//   handle:    conditional node handle (passed to cudaGraphSetConditional)
//   flag_slot: device pointer to a void* slot holding the address of the
//              user's qd.i32 counter ndarray
extern "C" __global__ void _qd_graph_do_while_cond(cudaGraphConditionalHandle handle, int32_t **flag_slot) {
  int32_t *flag = *flag_slot;
  cudaGraphSetConditional(handle, *flag != 0 ? 1u : 0u);
}
