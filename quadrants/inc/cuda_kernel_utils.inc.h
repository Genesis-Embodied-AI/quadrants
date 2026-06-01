extern "C" {

int thread_idx() {
  return 0;
}

// In-block thread index (workgroup-local) exposed as a separate symbol so it can be patched to a
// per-backend intrinsic without colliding with the internal C++ `thread_idx()` helper used
// throughout the runtime. Patched in `llvm_context.cpp` to `nvvm_read_ptx_sreg_tid_x` on CUDA and
// `amdgcn_workitem_id_x` on AMDGPU; surfaced to Python as `qd.simt.block.thread_idx()`.
int block_thread_idx() {
  return 0;
}

int warp_size() {
  return 32;
}

int warp_idx() {
  return thread_idx() % warp_size();
}

int block_idx() {
  return 0;
}

int block_dim() {
  return 0;
}

int grid_dim() {
  return 0;
}
}
