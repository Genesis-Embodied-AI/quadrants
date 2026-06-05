#pragma once

// Use relative path here for runtime compilation
#include "quadrants/inc/constants.h"
#include <cstdint>

#if defined(QD_RUNTIME_HOST)
namespace quadrants::lang {
#endif

struct LLVMRuntime;
// "RuntimeContext" holds necessary data for kernel body execution, such as a
// pointer to the LLVMRuntime struct, kernel arguments, and the thread id (if on
// CPU).
struct RuntimeContext {
  char *arg_buffer{nullptr};

  LLVMRuntime *runtime{nullptr};

  int32_t cpu_thread_id;

  // We move the pointer of result buffer from LLVMRuntime to RuntimeContext
  // because each real function need a place to store its result, but
  // LLVMRuntime is shared among functions. So we moved the pointer to
  // RuntimeContext which each function have one.
  uint64_t *result_buffer;

  // Set to 1 by quadrants_assert_format_ctx when a runtime assertion (e.g.
  // out-of-bounds check) fails on CPU.  The codegen emits an early return
  // after each assert call when this is set, and the task runner breaks out
  // of its loop.
  int32_t cpu_assert_failed{0};

  // qd.checkpoint() GPU-side gating: device pointers (NOT host pointers) to single int32 scalars
  // the cached graph allocates per launch. Used by the codegen-emitted prologue at the start of
  // every cp_id >= 0 body kernel:
  //   if (checkpoint_resume_point_ptr && *checkpoint_resume_point_ptr > cp_id) return;
  //   if (checkpoint_yield_signal_ptr && *checkpoint_yield_signal_ptr != -1) return;
  // The prologue is the gating mechanism on pre-Hopper CUDA (which has no conditional-graph-
  // node support); on SM 9.0+ the conditional gate already prevents the body kernel from
  // launching, so the prologue is dead code in the common path but stays in for correctness on
  // the rare overlapping-gate case. Null on kernels without checkpoints, and on all non-CUDA
  // backends (CPU does host-branch gating before the launch; Vulkan / Metal use indirect-dispatch
  // gating via SPIR-V gate shaders that don't go through RuntimeContext).
  int32_t *checkpoint_resume_point_ptr{nullptr};
  int32_t *checkpoint_yield_signal_ptr{nullptr};
};

#if defined(QD_RUNTIME_HOST)
}  // namespace quadrants::lang
#endif
