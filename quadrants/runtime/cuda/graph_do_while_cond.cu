// Source for the graph_do_while condition kernels.
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

// Condition kernel for graph_do_while bodies that also contain `qd.checkpoint(yield_on=...)`
// blocks. Same indirection trick on the user counter; additionally checks the framework's
// `yield_signal` scalar (-1 = no yield this iteration, otherwise the cp_id of the first
// checkpoint that fired its yield_on). On yield we want to exit the WHILE immediately --
// otherwise the body would re-enter, run the unconditional outer-chain tasks again, then have
// every checkpoint skip (because resume_point was bumped to INT_MAX by the yield-check kernel)
// and the counter would never decrement.
//
// Parameters:
//   handle:        conditional node handle (passed to cudaGraphSetConditional)
//   flag_slot:     device pointer to a void* slot holding the address of the user's qd.i32
//                  counter ndarray (same convention as `_qd_graph_do_while_cond`)
//   yield_signal:  device pointer to the framework's int32 yield_signal scalar; -1 means
//                  no yield, anything else means a yield occurred this iteration
extern "C" __global__ void _qd_graph_do_while_cond_with_yield(cudaGraphConditionalHandle handle,
                                                              int32_t **flag_slot,
                                                              int32_t *yield_signal) {
  int32_t *flag = *flag_slot;
  unsigned int cont = ((*flag != 0) && (*yield_signal == -1)) ? 1u : 0u;
  cudaGraphSetConditional(handle, cont);
}
