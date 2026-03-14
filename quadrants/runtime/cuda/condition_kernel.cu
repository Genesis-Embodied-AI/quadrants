// Condition kernel for graph_do_while conditional nodes.
//
// Reads a device-side int32 flag and calls cudaGraphSetConditional to
// control whether the conditional while node continues iterating.
//
// To regenerate the pre-built fatbin from this source, run:
//   scripts/build_condition_kernel_fatbin.sh

extern "C" __global__ void _qd_graph_do_while_cond(
    unsigned long long handle, int *flag_ptr) {
  int val = *flag_ptr;
  cudaGraphSetConditional(handle, val != 0 ? 1 : 0);
}
