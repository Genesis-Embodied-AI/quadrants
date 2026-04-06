// Source for the graph_do_while condition kernel.
//
// This file is the source of truth for the PTX string kConditionKernelPTX
// in graph_manager.cpp. After editing, regenerate the PTX and paste
// the entire output into the R"PTX(...)PTX" literal:
//
//   nvcc -ptx -arch=sm_90 -rdc=true graph_do_while_cond.cu \
//        && cat graph_do_while_cond.ptx

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
extern "C" __global__ void _qd_graph_do_while_cond(
    cudaGraphConditionalHandle handle,
    int32_t **flag_slot) {
  int32_t *flag = *flag_slot;
  cudaGraphSetConditional(handle, *flag != 0 ? 1u : 0u);
}
