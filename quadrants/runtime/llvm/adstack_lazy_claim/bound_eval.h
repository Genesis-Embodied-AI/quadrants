// Stage A of the LLVM sparse-adstack-heap lazy-claim pipeline: per-launch buffer publish + bound-expression
// evaluation. Allocates / clears the per-task lazy-claim arrays (`adstack_row_counters[num_tasks]` for the
// LCA-block atomic-rmw target, `adstack_bound_row_capacities[num_tasks]` for the codegen-emitted bounds
// clamp), then per task evaluates the captured `StaticAdStackBoundExpr` over `[0, length)` and publishes the
// gate-passing count into the per-task capacity slot. CPU walks the gating field on the host directly; CUDA
// / AMDGPU dispatch a single-thread device-side reducer (`runtime_eval_static_bound_count` in
// `runtime_module/runtime.cpp`). Captured `MaxOverRange` leaves are resolved up front by
// `dispatch_max_reducers_for_tasks` so the per-task sizer in Stage B sees them as `Const` substitutions.
//
// Caller responsibility (in `kernel_launcher.cpp` for each arch): invoke `publish_adstack_lazy_claim_buffers`
// once per kernel-launch before the first task dispatches, then per task call either
// `publish_per_task_bound_count_cpu` or `publish_per_task_bound_count_device` (arch-dispatched). Tasks
// without a captured `bound_expr` have those calls early-return with the inert UINT32_MAX sentinel that
// `publish_adstack_lazy_claim_buffers` wrote.
//
// All entry points are member methods of `LlvmRuntimeExecutor` and stay declared in
// `quadrants/runtime/llvm/llvm_runtime_executor.h`. This header carries only the file-private helpers
// shared between the stage's translation unit and (potentially) future cross-stage callers.

#pragma once

#include <limits>

#include "quadrants/ir/static_adstack_bound_reducer_device.h"
#include "quadrants/ir/stmt_op_types.h"

namespace quadrants::lang {

namespace {

// Encode the captured `BinaryOpType` (stored as int in `cmp_op`) and evaluate against typed operands. Mirrors the
// SPIR-V reducer's `OpSwitch` over the same encoding.
template <typename T>
inline bool eval_cmp(int cmp_op, T lhs, T rhs) {
  switch (static_cast<BinaryOpType>(cmp_op)) {
    case BinaryOpType::cmp_lt:
      return lhs < rhs;
    case BinaryOpType::cmp_le:
      return lhs <= rhs;
    case BinaryOpType::cmp_gt:
      return lhs > rhs;
    case BinaryOpType::cmp_ge:
      return lhs >= rhs;
    case BinaryOpType::cmp_eq:
      return lhs == rhs;
    case BinaryOpType::cmp_ne:
      return lhs != rhs;
    default:
      return false;
  }
}

// Encode the captured `BinaryOpType` into the 0-5 numeric range the LLVM device reducer's switch consumes. Mirrors the
// SPIR-V reducer's `encode_cmp_op` mapping at `quadrants/runtime/gfx/adstack_bound_reducer_launch.cpp`.
inline uint32_t encode_cmp_op_for_llvm_reducer(int captured_cmp_op) {
  switch (static_cast<BinaryOpType>(captured_cmp_op)) {
    case BinaryOpType::cmp_lt:
      return kLlvmReducerCmpLt;
    case BinaryOpType::cmp_le:
      return kLlvmReducerCmpLe;
    case BinaryOpType::cmp_gt:
      return kLlvmReducerCmpGt;
    case BinaryOpType::cmp_ge:
      return kLlvmReducerCmpGe;
    case BinaryOpType::cmp_eq:
      return kLlvmReducerCmpEq;
    case BinaryOpType::cmp_ne:
      return kLlvmReducerCmpNe;
    default:
      return std::numeric_limits<uint32_t>::max();
  }
}

}  // namespace

}  // namespace quadrants::lang
