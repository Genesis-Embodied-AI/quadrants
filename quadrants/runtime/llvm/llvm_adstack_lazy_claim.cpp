// Static-IR-bound sparse-adstack-heap reducer dispatch + lazy-claim buffer plumbing + split-heap grow-on-demand for
// LLVM backends (CPU / CUDA / AMDGPU). Extracted out of `llvm_runtime_executor.cpp` for the same reason the SPIR-V
// counterpart `quadrants/runtime/gfx/adstack_bound_reducer_launch.cpp` is - keep `LlvmRuntimeExecutor`'s body
// focused on runtime-init / SNode / kernel-launch plumbing that is not tied to the bound-reducer feature.
//
// Methods landing here all share the same triple of responsibilities, gated on the captured `bound_expr` field of
// `AdStackSizingInfo`:
//   1. Allocate / clear the per-task lazy-claim arrays (`adstack_row_counters[num_tasks]` for the LCA-block
//      atomic-rmw target, `adstack_bound_row_capacities[num_tasks]` for the codegen-emitted bounds clamp).
//   2. Evaluate the captured `StaticAdStackBoundExpr` over `[0, length)` and publish the gate-passing count into
//      the per-task capacity slot. CPU walks the gating field on the host directly; CUDA / AMDGPU dispatch a
//      single-thread device-side reducer (`runtime_eval_static_bound_count` in `runtime_module/runtime.cpp`).
//   3. Size the float / int adstack heaps from the published count via `ensure_adstack_heap_float` /
//      `ensure_adstack_heap_int` so each heap holds exactly `count * stride` bytes per dispatch instead of the
//      dispatched-threads worst case. The split-heap field-of-LLVMRuntime addresses are cached on first grow by
//      either `_float` or `_int` (the `runtime_get_adstack_split_heap_field_ptrs` getter returns all four in
//      fixed slot order).
//
// All methods (and the two anonymous-namespace helpers) are conditional on at least one task in the kernel having
// a captured `bound_expr`; on kernels without one, or on the `cfg_optimization=False` cache-miss path that did not
// capture a gate, the methods early-return UINT32_MAX (capacity stays at the inert sentinel
// `publish_adstack_lazy_claim_buffers` wrote) and the dispatched-threads worst-case heap sizing remains in force.
//
// Caller responsibility (in `kernel_launcher.cpp` for each arch): invoke `publish_adstack_lazy_claim_buffers` once
// per kernel-launch before the first task dispatches, then per task call either `publish_per_task_bound_count_cpu`
// or `publish_per_task_bound_count_device` (arch-dispatched), then `ensure_per_task_float_heap_post_reducer`. Tasks
// without a captured `bound_expr` have those calls early-return.

#include "quadrants/runtime/llvm/llvm_runtime_executor.h"
#include "quadrants/program/adstack_size_expr_eval.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "quadrants/ir/static_adstack_bound_reducer_device.h"
#include "quadrants/ir/stmt_op_types.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program_impl.h"
#include "quadrants/rhi/llvm/llvm_device.h"

#include "quadrants/platform/cuda/detect_cuda.h"
#include "quadrants/rhi/cuda/cuda_driver.h"
#if defined(QD_WITH_CUDA)
#include "quadrants/rhi/cuda/cuda_context.h"
#endif

#include "quadrants/platform/amdgpu/detect_amdgpu.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"
#if defined(QD_WITH_AMDGPU)
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#endif

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
uint32_t encode_cmp_op_for_llvm_reducer(int captured_cmp_op) {
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

uint32_t LlvmRuntimeExecutor::publish_per_task_bound_count_cpu(std::size_t task_index,
                                                               const AdStackSizingInfo &ad_stack,
                                                               std::size_t length,
                                                               LaunchContextBuilder *ctx) {
  // Default to UINT32_MAX (no clamp); only override on a successful host evaluation. The codegen-emitted bounds clamp
  // at the float LCA-block claim site stays inert when the slot holds UINT32_MAX, so this fall-through is a no-op that
  // preserves the existing behaviour.
  if (config_.arch != Arch::x64 && config_.arch != Arch::arm64) {
    return std::numeric_limits<uint32_t>::max();
  }
  if (!ad_stack.bound_expr.has_value()) {
    return std::numeric_limits<uint32_t>::max();
  }
  const auto &be = ad_stack.bound_expr.value();

  // Resolve the per-iteration field address. Two source kinds (mirrors the device-side reducer in
  // `runtime_eval_static_bound_count`):
  //   * NdArray: walk `arg_buffer + data_ptr_byte_off` to fetch the ndarray's data pointer; the gating field
  //     is then `data_ptr[i]` for `i in [0, length)`. On CPU `arg_buffer` lives in host memory, so the deref is direct.
  //   * SNode: walk `runtime->roots[snode_root_id] + snode_byte_base_offset + i * snode_byte_cell_stride`
  //     for `i in [0, length)`. The byte offset / cell stride were resolved by the codegen-time SNode descriptor
  //     resolver (via `compile_snode_structs`); `runtime->roots` is host-resident on CPU and reachable through the
  //     `LLVMRuntime_get_roots` STRUCT_FIELD_ARRAY getter.
  // Without the SNode arm, kernels with a captured SNode-backed bound_expr leave the capacity slot at UINT32_MAX (the
  // `publish_adstack_lazy_claim_buffers` default), `ensure_per_task_float_heap_post_reducer` sizes the float heap at
  // the worst-case num_threads count, and the codegen-emitted clamp goes inert -exactly the regression a `for i in
  // selector: if selector[i] > eps:` SNode-gated reverse kernel hits when the float adstack heap can only hold
  // `num_cpu_threads` rows but the LCA-block atomic-rmw fires once per gated iteration.
  using FSK = StaticAdStackBoundExpr::FieldSourceKind;
  if (be.field_source_kind != FSK::NdArray && be.field_source_kind != FSK::SNode) {
    return std::numeric_limits<uint32_t>::max();
  }

  const char *field_base = nullptr;
  std::size_t field_stride_bytes = 0;
  if (be.field_source_kind == FSK::NdArray) {
    if (ctx == nullptr || ctx->args_type == nullptr || ctx->get_context().arg_buffer == nullptr) {
      return std::numeric_limits<uint32_t>::max();
    }
    std::vector<int> indices = be.ndarray_arg_id;
    indices.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
    std::size_t data_ptr_byte_off = ctx->args_type->get_element_offset(indices);
    const char *arg_buffer = static_cast<const char *>(ctx->get_context().arg_buffer);
    void *data_ptr = *reinterpret_cast<void *const *>(arg_buffer + data_ptr_byte_off);
    if (data_ptr == nullptr) {
      return std::numeric_limits<uint32_t>::max();
    }
    field_base = static_cast<const char *>(data_ptr);
    field_stride_bytes = be.field_dtype_is_double ? sizeof(double) : sizeof(int32_t);  // f32 / i32 = 4 B, f64 = 8 B.
  } else {
    // SNode-backed source: query the host-resident `runtime->roots[snode_root_id]` pointer through the
    // STRUCT_FIELD_ARRAY getter; on CPU this is an in-process call (no DtoH stage) and returns the dense root buffer
    // base address directly.
    if (be.snode_root_id < 0 || llvm_runtime_ == nullptr || result_buffer_cache_ == nullptr) {
      return std::numeric_limits<uint32_t>::max();
    }
    // `RUNTIME_STRUCT_FIELD_ARRAY(LLVMRuntime, roots)` defines `runtime_LLVMRuntime_get_roots(LLVMRuntime *runtime,
    // LLVMRuntime *s, int i)` (the macro takes a struct-of-interest argument distinct from the runtime context, but for
    // fields of `LLVMRuntime` itself the two pointers are the same). `runtime_query` auto-prepends `llvm_runtime_` as
    // the first arg, so we pass `(llvm_runtime_, root_id)` to make the call resolve to the 3-arg signature
    // `(llvm_runtime_, llvm_runtime_, root_id)`. Mirrors the `node_allocators` call site a few hundred lines above.
    void *root_ptr =
        runtime_query<void *>("LLVMRuntime_get_roots", result_buffer_cache_, llvm_runtime_, be.snode_root_id);
    if (root_ptr == nullptr) {
      return std::numeric_limits<uint32_t>::max();
    }
    field_base = static_cast<const char *>(root_ptr) + be.snode_byte_base_offset;
    field_stride_bytes = static_cast<std::size_t>(be.snode_byte_cell_stride);
  }

  // Walk `[0, length)` evaluating the captured predicate on each thread's `field[i]`. The polarity bit selects
  // enter-on-true vs enter-on-false at the LCA's IfStmt; the count we publish is always the number of threads that
  // REACH the LCA, regardless of the gate orientation. f64 gates dispatch through the same float-source arm but read
  // the source as `double*` and compare against `literal_f64` so the f64 precision the user declared is preserved
  // end-to-end (narrowing the literal to f32 here would risk false-positive / negative counts on gates whose threshold
  // sits within the f32 representable gap).
  uint32_t count = 0;
  if (be.field_dtype_is_float) {
    if (be.field_dtype_is_double) {
      for (std::size_t i = 0; i < length; ++i) {
        const double v = *reinterpret_cast<const double *>(field_base + i * field_stride_bytes);
        const bool match = eval_cmp<double>(be.cmp_op, v, be.literal_f64);
        if (be.polarity ? match : !match) {
          ++count;
        }
      }
    } else {
      for (std::size_t i = 0; i < length; ++i) {
        const float v = *reinterpret_cast<const float *>(field_base + i * field_stride_bytes);
        const bool match = eval_cmp<float>(be.cmp_op, v, be.literal_f32);
        if (be.polarity ? match : !match) {
          ++count;
        }
      }
    }
  } else {
    for (std::size_t i = 0; i < length; ++i) {
      const int32_t v = *reinterpret_cast<const int32_t *>(field_base + i * field_stride_bytes);
      const bool match = eval_cmp<int32_t>(be.cmp_op, v, be.literal_i32);
      if (be.polarity ? match : !match) {
        ++count;
      }
    }
  }

  // Publish the count into `runtime->adstack_bound_row_capacities[task_index]` so the codegen-emitted bounds clamp at
  // the float LCA-block claim site reads it back as the per-task capacity. Slot was reset to UINT32_MAX by
  // `publish_adstack_lazy_claim_buffers`; this overwrite tightens it to the real count.
  if (runtime_adstack_bound_row_capacities_field_ptr_ == nullptr || adstack_bound_row_capacities_alloc_ == nullptr) {
    return count;
  }
  void *bound_capacities_dev_ptr = get_device_alloc_info_ptr(*adstack_bound_row_capacities_alloc_);
  // CPU only: write directly into the host-resident array.
  uint32_t *slots = static_cast<uint32_t *>(bound_capacities_dev_ptr);
  slots[task_index] = count;
  return count;
}

void LlvmRuntimeExecutor::publish_per_task_bound_count_device(std::size_t task_index,
                                                              const AdStackSizingInfo &ad_stack,
                                                              std::size_t length,
                                                              LaunchContextBuilder *ctx,
                                                              void *device_runtime_context_ptr) {
  // Only fires for CUDA / AMDGPU; CPU goes through `publish_per_task_bound_count_cpu`. Bail when the task did not
  // capture a bound_expr (no clamp needed - the slot stays at the UINT32_MAX default that
  // `publish_adstack_lazy_claim_buffers` wrote). Both ndarray and SNode source kinds are dispatched through the same
  // params blob; the device-side reducer selects between them via `field_source_is_snode`.
  if (config_.arch != Arch::cuda && config_.arch != Arch::amdgpu) {
    return;
  }
  if (!ad_stack.bound_expr.has_value()) {
    return;
  }
  const auto &be = ad_stack.bound_expr.value();
  const bool is_snode_source = be.field_source_kind == StaticAdStackBoundExpr::FieldSourceKind::SNode;
  if (ctx == nullptr || ctx->args_type == nullptr) {
    return;
  }
  const uint32_t cmp_op_encoded = encode_cmp_op_for_llvm_reducer(be.cmp_op);
  if (cmp_op_encoded == std::numeric_limits<uint32_t>::max()) {
    return;  // unrecognised comparison op (the IR pattern matcher should have rejected it earlier)
  }

  // Fill the device-side params struct on the host. Threshold bits live as the same u32 the runtime function bitcasts
  // back; we copy whichever underlying integer or float value the analysis captured. The two source shapes (ndarray +
  // SNode) share the comparison fields and differ only in which trailing fields the reducer reads (`arg_word_offset`
  // for ndarray, `snode_root_id` + `snode_byte_*` for SNode); host-side we populate the matching pair and zero out the
  // other.
  LlvmAdStackBoundReducerDeviceParams params{};
  params.task_index = static_cast<uint32_t>(task_index);
  params.length = static_cast<uint32_t>(is_snode_source ? be.snode_iter_count : length);
  params.cmp_op = cmp_op_encoded;
  params.field_dtype_is_float = be.field_dtype_is_float ? 1u : 0u;
  params.field_dtype_is_double = be.field_dtype_is_double ? 1u : 0u;
  params.polarity = be.polarity ? 1u : 0u;
  if (be.field_dtype_is_double) {
    // Pack the f64 threshold's 64-bit pattern into the (lo, hi) u32 pair the reducer reassembles.
    uint64_t bits64 = 0;
    std::memcpy(&bits64, &be.literal_f64, sizeof(uint64_t));
    params.threshold_bits = static_cast<uint32_t>(bits64 & 0xFFFFFFFFu);
    params.threshold_bits_high = static_cast<uint32_t>(bits64 >> 32);
  } else if (be.field_dtype_is_float) {
    std::memcpy(&params.threshold_bits, &be.literal_f32, sizeof(uint32_t));
  } else {
    params.threshold_bits = static_cast<uint32_t>(be.literal_i32);
  }
  params.field_source_is_snode = is_snode_source ? 1u : 0u;
  if (is_snode_source) {
    params.arg_word_offset = 0;
    params.snode_root_id = static_cast<uint32_t>(be.snode_root_id);
    params.snode_byte_base_offset = be.snode_byte_base_offset;
    params.snode_byte_cell_stride = be.snode_byte_cell_stride;
  } else {
    // Resolve the ndarray data pointer's word offset within the kernel arg buffer. Same path the SPIR-V reducer and the
    // CPU host-eval use; bytes -> words for the reducer's `arg_buffer_u32[arg_word_offset]` indexing.
    std::vector<int> indices = be.ndarray_arg_id;
    indices.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
    std::size_t data_ptr_byte_off = ctx->args_type->get_element_offset(indices);
    if (data_ptr_byte_off % sizeof(uint32_t) != 0) {
      return;  // misaligned offset; the reducer's u32-word indexing would lose bits.
    }
    params.arg_word_offset = static_cast<uint32_t>(data_ptr_byte_off / sizeof(uint32_t));
    params.snode_root_id = 0;
    params.snode_byte_base_offset = 0;
    params.snode_byte_cell_stride = 0;
  }

  // Lazy-allocate the device-side params scratch buffer the first time a bound_expr task fires; reuse for subsequent
  // tasks across kernels. Sized for one struct (the reducer is single-task per call); a future optimisation could pack
  // multiple tasks' params into one buffer and dispatch them in a single launch.
  const std::size_t needed_bytes = sizeof(LlvmAdStackBoundReducerDeviceParams);
  if (needed_bytes > adstack_bound_reducer_params_capacity_) {
    Device::AllocParams alloc_params{};
    alloc_params.size = std::max<std::size_t>(needed_bytes, 2 * adstack_bound_reducer_params_capacity_);
    alloc_params.host_read = false;
    alloc_params.host_write = true;
    alloc_params.export_sharing = false;
    alloc_params.usage = AllocUsage::Storage;
    DeviceAllocation new_alloc;
    RhiResult res = llvm_device()->allocate_memory(alloc_params, &new_alloc);
    QD_ERROR_IF(res != RhiResult::success,
                "Failed to allocate {} bytes for adstack bound reducer params buffer (err: {})", alloc_params.size,
                int(res));
    adstack_bound_reducer_params_alloc_ = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));
    adstack_bound_reducer_params_capacity_ = alloc_params.size;
  }
  void *params_dev_ptr = get_device_alloc_info_ptr(*adstack_bound_reducer_params_alloc_);

  // h2d the params struct into the device buffer.
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().memcpy_host_to_device(params_dev_ptr, &params, needed_bytes);
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memcpy_host_to_device(params_dev_ptr, &params, needed_bytes);
#else
    QD_NOT_IMPLEMENTED;
#endif
  }

  // Dispatch the runtime reducer function: single-threaded device-side walk that reads `ctx->arg_buffer` (the
  // device-mirror the launcher staged) and writes the count into `runtime->adstack_bound_row_capacities[task_index]`.
  // Pass the device-side `RuntimeContext` pointer the same way the size-expr sizer does so the function can deref
  // `ctx->arg_buffer` on-device.
  auto *const runtime_jit = get_runtime_jit_module();
  void *runtime_context_ptr_for_reducer =
      device_runtime_context_ptr != nullptr ? device_runtime_context_ptr : static_cast<void *>(&ctx->get_context());
  runtime_jit->call<void *, void *, void *>("runtime_eval_static_bound_count", llvm_runtime_,
                                            runtime_context_ptr_for_reducer, params_dev_ptr);
}

void LlvmRuntimeExecutor::ensure_adstack_heap_int(std::size_t needed_bytes) {
  if (needed_bytes == 0 || needed_bytes <= adstack_heap_size_int_) {
    return;
  }
  std::size_t new_size = std::max(needed_bytes, std::size_t(2) * adstack_heap_size_int_);

  Device::AllocParams params{};
  params.size = new_size;
  params.host_read = false;
  params.host_write = false;
  params.export_sharing = false;
  params.usage = AllocUsage::Storage;
  DeviceAllocation new_alloc;
  RhiResult res = llvm_device()->allocate_memory(params, &new_alloc);
  QD_ERROR_IF(res != RhiResult::success,
              "Failed to allocate {} bytes for the adstack int heap (err: {}). Consider lowering "
              "`ad_stack_size` or the per-kernel reverse-mode adstack count.",
              new_size, int(res));
  void *new_ptr = get_device_alloc_info_ptr(new_alloc);
  auto new_guard = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));

  // The split-heap field-of-LLVMRuntime addresses are cached together by `ensure_adstack_heap_float` on its first grow
  // (the same `runtime_get_adstack_split_heap_field_ptrs` getter returns all four addresses - float-buffer, float-size,
  // int-buffer, int-size - in fixed slot order). On a fresh executor where this is the very first split-heap call,
  // resolve the addresses here so we can publish independently of the float heap path.
  if (runtime_adstack_heap_buffer_int_field_ptr_ == nullptr) {
    auto *const runtime_jit = get_runtime_jit_module();
    runtime_jit->call<void *>("runtime_get_adstack_split_heap_field_ptrs", llvm_runtime_);
    runtime_adstack_heap_buffer_float_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id, result_buffer_cache_));
    runtime_adstack_heap_size_float_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 1, result_buffer_cache_));
    runtime_adstack_heap_buffer_int_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 2, result_buffer_cache_));
    runtime_adstack_heap_size_int_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 3, result_buffer_cache_));
  }
  uint64 size_u64 = static_cast<uint64>(new_size);
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_buffer_int_field_ptr_, &new_ptr,
                                                     sizeof(void *));
    CUDADriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_size_int_field_ptr_, &size_u64,
                                                     sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_buffer_int_field_ptr_, &new_ptr,
                                                       sizeof(void *));
    AMDGPUDriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_size_int_field_ptr_, &size_u64,
                                                       sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else {
    *reinterpret_cast<void **>(runtime_adstack_heap_buffer_int_field_ptr_) = new_ptr;
    *reinterpret_cast<uint64 *>(runtime_adstack_heap_size_int_field_ptr_) = size_u64;
  }

  adstack_heap_alloc_int_ = std::move(new_guard);
  adstack_heap_size_int_ = new_size;
}

void LlvmRuntimeExecutor::ensure_per_task_float_heap_post_reducer(std::size_t task_index,
                                                                  const AdStackSizingInfo &ad_stack,
                                                                  std::size_t num_threads,
                                                                  LaunchContextBuilder *ctx) {
  // Skip when the task has no float heap need (no f32 allocas, or analysis didn't capture a gate so we wouldn't have
  // routed it through the lazy float path on the codegen side).
  if (!ad_stack.bound_expr.has_value() || ad_stack.per_thread_stride_float == 0) {
    return;
  }

  // Read the per-task count the reducer published. On CPU the capacity buffer is host-resident; on CUDA / AMDGPU it's
  // device memory and the read is a small (4-byte) DtoH per task. Cost is dominated by the actual main kernel.
  uint32_t count = std::numeric_limits<uint32_t>::max();
  if (adstack_bound_row_capacities_alloc_) {
    void *capacities_dev_ptr = get_device_alloc_info_ptr(*adstack_bound_row_capacities_alloc_);
    char *slot_ptr = static_cast<char *>(capacities_dev_ptr) + task_index * sizeof(uint32_t);
    if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
      CUDADriver::get_instance().memcpy_device_to_host(&count, slot_ptr, sizeof(uint32_t));
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
      AMDGPUDriver::get_instance().memcpy_device_to_host(&count, slot_ptr, sizeof(uint32_t));
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else {
      count = *reinterpret_cast<const uint32_t *>(slot_ptr);
    }
  }

  // Floor at 1 row when the captured count is zero (no thread passed the gate this launch). The codegen-emitted bounds
  // clamp keeps `claimed_row` in [0, count-1] so threads that miss the gate never reach the LCA-block claim - the heap
  // row stays unused. A 1-row allocation is cheap and keeps the heap pointer non-null. Clip by the captured
  // compile-time loop trip count when known: each iteration claims at most one row at the LCA-block (one `atomic_add`
  // per gating iteration), so the heap needs at most `loop_iter_static` rows regardless of how many cells of an
  // oversized gating SNode the reducer counted. The analyzer leaves `loop_iter_static == 0` for runtime-bounded loops
  // and for CPU LLVM tasks whose `[begin_value, end_value)` is a post-chunking subrange (the unclipped reducer count is
  // the right upper bound there).
  std::size_t effective_rows =
      (count == std::numeric_limits<uint32_t>::max()) ? num_threads : std::max<std::size_t>(count, 1);
  if (count != std::numeric_limits<uint32_t>::max() && ad_stack.bound_expr.has_value()) {
    // Shared with the SPIR-V launcher: see `clip_effective_rows_by_loop_trip_count` in
    // `program/adstack_size_expr_eval.cpp`. LLVM dispatches one thread per loop iteration without the
    // SPIR-V dispatch-cap-driven serialisation, so pass `numeric_limits::max()` to disable the
    // dispatched-threads ceiling - any positive trip-count value is a sound upper bound on row claims
    // here. `numeric_limits<size_t>::max()` is the ceiling sentinel `clip_effective_rows_by_loop_trip_count`
    // documents.
    Program *prog = (program_impl_ != nullptr) ? program_impl_->program : nullptr;
    clip_effective_rows_by_loop_trip_count(effective_rows, *ad_stack.bound_expr,
                                           std::numeric_limits<std::size_t>::max(), prog, ctx);
  }
  // Read back the per-thread float stride (in bytes) that `publish_adstack_metadata` published into
  // `runtime->adstack_per_thread_stride_float`. `AdStackSizingInfo::per_thread_stride_float` from the analysis pre-pass
  // is in entry-count units (`2 * max_size`), not bytes, and would massively undersize the heap.
  uint64_t stride_float_bytes_u64 = 0;
  if (runtime_adstack_stride_float_field_ptr_ != nullptr) {
    if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
      CUDADriver::get_instance().memcpy_device_to_host(&stride_float_bytes_u64, runtime_adstack_stride_float_field_ptr_,
                                                       sizeof(uint64_t));
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
      AMDGPUDriver::get_instance().memcpy_device_to_host(&stride_float_bytes_u64,
                                                         runtime_adstack_stride_float_field_ptr_, sizeof(uint64_t));
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else {
      stride_float_bytes_u64 = *reinterpret_cast<const uint64_t *>(runtime_adstack_stride_float_field_ptr_);
    }
  }
  const std::size_t needed_bytes = effective_rows * static_cast<std::size_t>(stride_float_bytes_u64);
  // `QD_DEBUG_ADSTACK=1` opt-in diagnostic. Persistent so memory regressions can be debugged without re-instrumenting.
  if (std::getenv("QD_DEBUG_ADSTACK")) {
    const char *src = (count == std::numeric_limits<uint32_t>::max())
                          ? "worst_case_num_threads"
                          : (count == 0 ? "reducer_zero_floored" : "reducer_count");
    std::fprintf(stderr,
                 "[adstack_heap] arch=llvm task_idx=%zu kind=F src=%s effective_rows=%zu stride=%llu "
                 "required_bytes=%zu (%.2f MB)\n",
                 task_index, src, effective_rows, static_cast<unsigned long long>(stride_float_bytes_u64), needed_bytes,
                 double(needed_bytes) / (1024.0 * 1024.0));
    std::fflush(stderr);
  }
  ensure_adstack_heap_float(needed_bytes);
}

void LlvmRuntimeExecutor::publish_adstack_lazy_claim_buffers(std::size_t num_tasks) {
  if (num_tasks == 0) {
    return;
  }
  // Cache the field-of-LLVMRuntime addresses for the row counter / bound row capacity array pointers. Resolved once per
  // program lifetime; subsequent grows write the new array pointers directly to the cached addresses.
  if (runtime_adstack_row_counters_field_ptr_ == nullptr) {
    auto *const runtime_jit = get_runtime_jit_module();
    runtime_jit->call<void *>("runtime_get_adstack_lazy_claim_field_ptrs", llvm_runtime_);
    runtime_adstack_row_counters_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id, result_buffer_cache_));
    runtime_adstack_bound_row_capacities_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 1, result_buffer_cache_));
  }

  auto grow_to = [&](DeviceAllocationUnique &alloc, std::size_t capacity_u32) {
    Device::AllocParams params{};
    params.size = capacity_u32 * sizeof(uint32_t);
    params.host_read = false;
    params.host_write = false;
    params.export_sharing = false;
    params.usage = AllocUsage::Storage;
    DeviceAllocation new_alloc;
    RhiResult res = llvm_device()->allocate_memory(params, &new_alloc);
    QD_ERROR_IF(res != RhiResult::success, "Failed to allocate {} bytes for adstack lazy-claim array (err: {})",
                params.size, int(res));
    alloc = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));
  };

  bool grew = false;
  if (num_tasks > adstack_lazy_claim_capacity_) {
    std::size_t new_cap = std::max<std::size_t>(num_tasks, 2 * adstack_lazy_claim_capacity_);
    grow_to(adstack_row_counters_alloc_, new_cap);
    grow_to(adstack_bound_row_capacities_alloc_, new_cap);
    adstack_lazy_claim_capacity_ = new_cap;
    grew = true;
  }
  void *row_counters_dev_ptr = get_device_alloc_info_ptr(*adstack_row_counters_alloc_);
  void *bound_capacities_dev_ptr = get_device_alloc_info_ptr(*adstack_bound_row_capacities_alloc_);

  // After every grow, publish the new array pointers into the runtime so the codegen-emitted GEPs
  // (`runtime->adstack_row_counters[task_codegen_id]` and `runtime->adstack_bound_row_capacities[task_codegen_id]`)
  // resolve against the live allocations. Skipped between grows because the cached field address holds the same pointer
  // value.
  auto copy_h2d = [&](void *dst, const void *src, std::size_t bytes) {
    if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
      CUDADriver::get_instance().memcpy_host_to_device(dst, const_cast<void *>(src), bytes);
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
      AMDGPUDriver::get_instance().memcpy_host_to_device(dst, const_cast<void *>(src), bytes);
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else {
      std::memcpy(dst, src, bytes);
    }
  };
  if (grew) {
    copy_h2d(runtime_adstack_row_counters_field_ptr_, &row_counters_dev_ptr, sizeof(void *));
    copy_h2d(runtime_adstack_bound_row_capacities_field_ptr_, &bound_capacities_dev_ptr, sizeof(void *));
  }

  // Per-launch reset: zero the counter slots (each task's LCA-block atomic-rmw add starts from 0 and accumulates its
  // own claims) and write UINT32_MAX into the capacity slots so the codegen-emitted bounds clamp is inert unless a
  // later reducer dispatch overrides slots with tighter counts. Memset rather than per-slot store: the host pays one
  // O(num_tasks) buffer fill per kernel-launch, regardless of arch.
  std::vector<uint32_t> zero_buf(num_tasks, 0u);
  std::vector<uint32_t> uint_max_buf(num_tasks, std::numeric_limits<uint32_t>::max());
  copy_h2d(row_counters_dev_ptr, zero_buf.data(), num_tasks * sizeof(uint32_t));
  copy_h2d(bound_capacities_dev_ptr, uint_max_buf.data(), num_tasks * sizeof(uint32_t));
}

void LlvmRuntimeExecutor::ensure_adstack_heap_float(std::size_t needed_bytes) {
  if (needed_bytes == 0 || needed_bytes <= adstack_heap_size_float_) {
    return;
  }
  // Mirror `ensure_adstack_heap`'s amortised-doubling growth and grow-on-demand semantics. The float heap is allocated
  // independently from the combined heap so a kernel with bound_expr tasks can shrink the combined slice to int-only
  // while still backing float allocas at `row_id_var * stride_float + float_offset`.
  std::size_t new_size = std::max(needed_bytes, std::size_t(2) * adstack_heap_size_float_);

  Device::AllocParams params{};
  params.size = new_size;
  params.host_read = false;
  params.host_write = false;
  params.export_sharing = false;
  params.usage = AllocUsage::Storage;
  DeviceAllocation new_alloc;
  RhiResult res = llvm_device()->allocate_memory(params, &new_alloc);
  QD_ERROR_IF(res != RhiResult::success,
              "Failed to allocate {} bytes for the adstack float heap (err: {}). Consider lowering "
              "`ad_stack_size` or the per-kernel reverse-mode adstack count.",
              new_size, int(res));
  void *new_ptr = get_device_alloc_info_ptr(new_alloc);
  auto new_guard = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));

  // Resolve and cache the field-of-LLVMRuntime addresses for the split-heap fields on first grow. The
  // `runtime_get_adstack_split_heap_field_ptrs` helper returns four addresses in fixed slot order: float-buffer-ptr,
  // float-size, int-buffer-ptr, int-size. We only consume the float pair here; the int half is reserved for a future
  // symmetric `ensure_adstack_heap_int` if it becomes useful (today the int allocas in bound_expr tasks ride the
  // combined heap with a smaller stride).
  if (runtime_adstack_heap_buffer_float_field_ptr_ == nullptr) {
    auto *const runtime_jit = get_runtime_jit_module();
    runtime_jit->call<void *>("runtime_get_adstack_split_heap_field_ptrs", llvm_runtime_);
    runtime_adstack_heap_buffer_float_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id, result_buffer_cache_));
    runtime_adstack_heap_size_float_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 1, result_buffer_cache_));
    runtime_adstack_heap_buffer_int_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 2, result_buffer_cache_));
    runtime_adstack_heap_size_int_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 3, result_buffer_cache_));
  }
  uint64 size_u64 = static_cast<uint64>(new_size);
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_buffer_float_field_ptr_, &new_ptr,
                                                     sizeof(void *));
    CUDADriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_size_float_field_ptr_, &size_u64,
                                                     sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_buffer_float_field_ptr_, &new_ptr,
                                                       sizeof(void *));
    AMDGPUDriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_size_float_field_ptr_, &size_u64,
                                                       sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else {
    *reinterpret_cast<void **>(runtime_adstack_heap_buffer_float_field_ptr_) = new_ptr;
    *reinterpret_cast<uint64 *>(runtime_adstack_heap_size_float_field_ptr_) = size_u64;
  }

  adstack_heap_alloc_float_ = std::move(new_guard);
  adstack_heap_size_float_ = new_size;
}

void LlvmRuntimeExecutor::check_adstack_overflow() {
  // Called from `synchronize()` on every sync so adstack overflow surfaces as a Python exception regardless of
  // `compile_config.debug`. The runtime / result buffer may not exist yet (e.g. a C++ test that constructs Program
  // without materializing the runtime and then triggers Program::finalize -> synchronize), so no-op in that case.
  if (llvm_runtime_ == nullptr || result_buffer_cache_ == nullptr) {
    return;
  }
  auto *runtime_jit_module = get_runtime_jit_module();
  runtime_jit_module->call<void *>("runtime_retrieve_and_reset_adstack_overflow", llvm_runtime_);
  auto flag = fetch_result<int64>(quadrants_result_buffer_error_id, result_buffer_cache_);
  if (flag != 0) {
    throw QuadrantsAssertionError(
        "Adstack overflow: a reverse-mode autodiff kernel pushed more elements than the adstack capacity "
        "allows. Raised at the next qd.sync() rather than at the offending kernel launch. The pre-pass "
        "resolved this alloca to a bound tighter than the actual runtime push count - either the enclosing "
        "loop shape is outside the current `SizeExpr` grammar (rewrite it, or extend the grammar), or the "
        "Bellman-Ford analyzer undercounted the forward-pass accumulation on this stack (file a bug with "
        "the kernel IR via `QD_DUMP_IR=1`).");
  }
}

std::size_t LlvmRuntimeExecutor::publish_adstack_metadata(const AdStackSizingInfo &ad_stack,
                                                          std::size_t num_threads,
                                                          LaunchContextBuilder *ctx,
                                                          void *device_runtime_context_ptr) {
  const auto n_stacks = ad_stack.allocas.size();
  if (n_stacks == 0 || num_threads == 0) {
    return 0;
  }
  auto align_up_8 = [](std::size_t n) -> std::size_t { return (n + 7u) & ~std::size_t{7u}; };
  // Allocate / grow the two device-side metadata arrays. Capacity is in u64 entries, kept at or above n_stacks.
  // On GPU these buffers are written exclusively by the device-side sizer kernel (`runtime_eval_adstack_size_expr`);
  // on CPU the host evaluator writes them directly via `std::memcpy`. Either way the pointers published into
  // `runtime->adstack_offsets` / `adstack_max_sizes` stay stable across launches unless we grow here.
  auto grow_to = [&](DeviceAllocationUnique &alloc, std::size_t capacity_u64) {
    Device::AllocParams params{};
    params.size = capacity_u64 * sizeof(uint64_t);
    params.host_read = false;
    params.host_write = false;
    params.export_sharing = false;
    params.usage = AllocUsage::Storage;
    DeviceAllocation new_alloc;
    RhiResult res = llvm_device()->allocate_memory(params, &new_alloc);
    QD_ERROR_IF(res != RhiResult::success, "Failed to allocate {} bytes for adstack metadata array (err: {})",
                params.size, int(res));
    alloc = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));
  };
  if (n_stacks > adstack_metadata_capacity_) {
    std::size_t new_cap = std::max<std::size_t>(n_stacks, 2 * adstack_metadata_capacity_);
    grow_to(adstack_offsets_alloc_, new_cap);
    grow_to(adstack_max_sizes_alloc_, new_cap);
    adstack_metadata_capacity_ = new_cap;
  }
  void *offsets_dev_ptr = get_device_alloc_info_ptr(*adstack_offsets_alloc_);
  void *max_sizes_dev_ptr = get_device_alloc_info_ptr(*adstack_max_sizes_alloc_);

  auto copy_h2d = [&](void *dst, const void *src, std::size_t bytes) {
    if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
      CUDADriver::get_instance().memcpy_host_to_device(dst, const_cast<void *>(src), bytes);
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
      AMDGPUDriver::get_instance().memcpy_host_to_device(dst, const_cast<void *>(src), bytes);
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else {
      std::memcpy(dst, src, bytes);
    }
  };
  auto copy_d2h = [&](void *dst, const void *src, std::size_t bytes) {
    if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
      CUDADriver::get_instance().memcpy_device_to_host(dst, const_cast<void *>(src), bytes);
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
      AMDGPUDriver::get_instance().memcpy_device_to_host(dst, const_cast<void *>(src), bytes);
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else {
      std::memcpy(dst, src, bytes);
    }
  };

  // Cache the runtime-field addresses on the first call; then publish the metadata-array pointers into the
  // runtime struct. The stride field is written by the sizer on GPU and by this function on CPU, so we cache the
  // address either way.
  if (runtime_adstack_stride_field_ptr_ == nullptr) {
    auto *const runtime_jit = get_runtime_jit_module();
    runtime_jit->call<void *>("runtime_get_adstack_metadata_field_ptrs", llvm_runtime_);
    // Slot order: combined-stride, offsets, max_sizes, float-stride, int-stride. Slots 0/1/2 keep the legacy ordering
    // for code paths that have not migrated to the split layout; slots 3/4 are new.
    runtime_adstack_stride_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id, result_buffer_cache_));
    runtime_adstack_offsets_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 1, result_buffer_cache_));
    runtime_adstack_max_sizes_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 2, result_buffer_cache_));
    runtime_adstack_stride_float_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 3, result_buffer_cache_));
    runtime_adstack_stride_int_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 4, result_buffer_cache_));
  }
  copy_h2d(runtime_adstack_offsets_field_ptr_, &offsets_dev_ptr, sizeof(void *));
  copy_h2d(runtime_adstack_max_sizes_field_ptr_, &max_sizes_dev_ptr, sizeof(void *));

  std::size_t stride = 0;
  const bool is_gpu_llvm = (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu);

  // Host-eval fast path. The on-device sizer kernel exists to handle one specific leaf, `ExternalTensorRead`,
  // whose ndarray data lives in GPU-private memory (`cudaMalloc` / `hipMalloc`, no UVA fallback) and thus
  // cannot be touched from the host. Every other SizeExpr leaf - `Const`, `BoundVariable`,
  // `ExternalTensorShape`, `FieldLoad` - is host-resolvable through the existing `evaluate_adstack_size_expr`
  // path, so when the kernel's SizeExprs are all `ExternalTensorRead`-free we can skip the encode + bytecode
  // h2d + sizer-kernel launch + d2h-stride pipeline entirely and write the metadata directly via `copy_h2d`.
  // On CUDA the saved `cuMemcpyDtoH` for the per-launch stride readback is the dominant cost: every reverse-
  // mode kernel launch in a 100-substep test paid one such synchronous DtoH each, and that compound stall
  // accounted for the bulk of the GPU launch overhead under adstack mode. The condition is computed once per
  // launch by scanning each stack's `nodes` vector for an `ExternalTensorRead` leaf; the scan is O(total
  // SizeExpr nodes), well below the cost of the cheapest h2d / d2h on any LLVM GPU backend.
  bool all_size_exprs_host_resolvable = true;
  for (std::size_t i = 0; i < n_stacks && all_size_exprs_host_resolvable; ++i) {
    if (i >= ad_stack.size_exprs.size()) {
      continue;
    }
    for (const auto &node : ad_stack.size_exprs[i].nodes) {
      if (static_cast<SizeExpr::Kind>(node.kind) == SizeExpr::Kind::ExternalTensorRead) {
        all_size_exprs_host_resolvable = false;
        break;
      }
    }
  }
  const bool use_host_eval = !is_gpu_llvm || all_size_exprs_host_resolvable;
  // Per-kind byte strides resolved either host-side (host-eval branch) or by reading back from the device runtime
  // struct after the sizer kernel ran (GPU branch). Used below to size the float / int heaps independently for the
  // unconditional split-heap layout.
  std::size_t stride_float_bytes = 0;
  std::size_t stride_int_bytes = 0;
  if (use_host_eval) {
    // CPU + GPU-without-ExternalTensorRead path: run the host evaluator directly. On CPU we use synchronous
    // `copy_h2d` (just `std::memcpy` for that arch), but on CUDA / AMDGPU we ship the same payload through
    // pinned-host memory via async `cuMemcpyHtoDAsync` / `hipMemcpyHtoDAsync` so the host returns immediately
    // after queueing the copies on the default stream and the subsequent main-kernel launch (also on the
    // default stream) stream-orders after the copies. The synchronous `cuMemcpyHtoD_v2` path used to block
    // the host on every one of the three writes we issue per launch; with thousands of reverse-mode launches
    // per `test_differentiable_rigid` run, those serial host stalls were a measurable fraction of wallclock.
    // `FieldLoad` is serviced by `SNodeRwAccessorsBank` regardless of arch.
    // Guard `program_impl_->program` lookups against the C++-only-tests setup where `program_impl_` itself is null;
    // the on-device branch below already does this and falls back to `max_size_compile_time`.
    Program *prog = (program_impl_ != nullptr) ? program_impl_->program : nullptr;
    std::vector<uint64_t> host_max_sizes(n_stacks);
    for (std::size_t i = 0; i < n_stacks; ++i) {
      const SerializedSizeExpr *expr = (i < ad_stack.size_exprs.size()) ? &ad_stack.size_exprs[i] : nullptr;
      int64_t v = -1;
      if (expr != nullptr && !expr->nodes.empty() && prog != nullptr) {
        v = evaluate_adstack_size_expr(*expr, prog, ctx);
      }
      if (v < 0) {
        v = static_cast<int64_t>(ad_stack.allocas[i].max_size_compile_time);
      }
      host_max_sizes[i] = static_cast<uint64_t>(std::max<int64_t>(v, 1));
    }
    // Unconditional split-heap layout: float allocas live at `host_offsets[i]` within the float-only slice (addressed
    // on the codegen side as `heap_float + row_id_var * stride_float + float_offset` for bound_expr tasks, or
    // `heap_float + linear_tid * stride_float + float_offset` for non-bound_expr tasks); int allocas live at
    // `host_offsets[i]` within the int-only slice (addressed as `heap_int + linear_tid * stride_int + int_offset`).
    // Same scheme regardless of `bound_expr` so the heap layout matches the SPIR-V backend's unconditional split into
    // `BufferType::AdStackHeapFloat` + `AdStackHeapInt`. The legacy combined-heap path is no longer used by the
    // codegen; the combined stride / heap fields stay in the LLVMRuntime struct only as a transitional fallback for
    // offline-cache-loaded kernels that predate the split, and the published `adstack_per_thread_stride` mirrors
    // `stride_int` so any such kernel sees the smaller int-only stride.
    std::vector<uint64_t> host_offsets(n_stacks);
    for (std::size_t i = 0; i < n_stacks; ++i) {
      const std::size_t step = align_up_8(sizeof(int64_t) + ad_stack.allocas[i].entry_size_bytes * host_max_sizes[i]);
      const bool is_float = ad_stack.allocas[i].heap_kind == AdStackAllocaInfo::HeapKind::Float;
      host_offsets[i] = is_float ? stride_float_bytes : stride_int_bytes;
      if (is_float) {
        stride_float_bytes += step;
      } else {
        stride_int_bytes += step;
      }
    }
    stride = stride_int_bytes;
    uint64_t stride_combined_u64 = static_cast<uint64_t>(stride);
    uint64_t stride_float_u64 = static_cast<uint64_t>(stride_float_bytes);
    uint64_t stride_int_u64 = static_cast<uint64_t>(stride_int_bytes);
    if (!is_gpu_llvm) {
      copy_h2d(offsets_dev_ptr, host_offsets.data(), n_stacks * sizeof(uint64_t));
      copy_h2d(max_sizes_dev_ptr, host_max_sizes.data(), n_stacks * sizeof(uint64_t));
      copy_h2d(runtime_adstack_stride_field_ptr_, &stride_combined_u64, sizeof(uint64_t));
      // Per-kind strides used by the split-heap codegen path; harmless when the codegen has not migrated yet (the
      // kernel reads only the combined stride). Skipped when the cache is empty (first launch on a stale executor
      // instance where `runtime_get_adstack_metadata_field_ptrs` populated only the legacy slots; the null check is
      // defensive - any host writing to `nullptr` would crash with no diagnostic).
      if (runtime_adstack_stride_float_field_ptr_ != nullptr) {
        copy_h2d(runtime_adstack_stride_float_field_ptr_, &stride_float_u64, sizeof(uint64_t));
      }
      if (runtime_adstack_stride_int_field_ptr_ != nullptr) {
        copy_h2d(runtime_adstack_stride_int_field_ptr_, &stride_int_u64, sizeof(uint64_t));
      }
    } else {
      // Five-block payload packed into the pinned-host scratch as `[stride_combined, stride_float, stride_int,
      // offsets[n_stacks], max_sizes[n_stacks]]`. Five async DMAs land on the matching device addresses; the driver's
      // H2D DMA engine reads from the pinned bytes at execution time, so we must not overwrite the scratch before all
      // copies have completed - hence the per-launch `event_record` after the last copy and the `event_synchronize` at
      // the top of the next launch.
      const std::size_t header_bytes = 3 * sizeof(uint64_t);
      const std::size_t array_bytes = n_stacks * sizeof(uint64_t);
      const std::size_t total_bytes = header_bytes + 2 * array_bytes;

      auto wait_pending = [this]() {
        if (!pinned_metadata_event_pending_) {
          return;
        }
#if defined(QD_WITH_CUDA)
        if (config_.arch == Arch::cuda) {
          CUDADriver::get_instance().event_synchronize(pinned_metadata_event_);
        }
#endif
#if defined(QD_WITH_AMDGPU)
        if (config_.arch == Arch::amdgpu) {
          AMDGPUDriver::get_instance().event_synchronize(pinned_metadata_event_);
        }
#endif
        pinned_metadata_event_pending_ = false;
      };

      // Grow / first-allocate the pinned host scratch and the per-launch completion event. Doubling growth
      // means the pinned alloc / free traffic is amortised to O(log peak_total_bytes) across a run.
      if (total_bytes > pinned_metadata_scratch_capacity_) {
        wait_pending();
        if (pinned_metadata_scratch_ != nullptr) {
#if defined(QD_WITH_CUDA)
          if (config_.arch == Arch::cuda) {
            CUDADriver::get_instance().mem_free_host(pinned_metadata_scratch_);
          }
#endif
#if defined(QD_WITH_AMDGPU)
          if (config_.arch == Arch::amdgpu) {
            AMDGPUDriver::get_instance().mem_free_host(pinned_metadata_scratch_);
          }
#endif
          pinned_metadata_scratch_ = nullptr;
        }
        std::size_t new_capacity = std::max<std::size_t>(total_bytes, 2 * pinned_metadata_scratch_capacity_);
#if defined(QD_WITH_CUDA)
        if (config_.arch == Arch::cuda) {
          CUDADriver::get_instance().mem_alloc_host(&pinned_metadata_scratch_, new_capacity);
        }
#endif
#if defined(QD_WITH_AMDGPU)
        if (config_.arch == Arch::amdgpu) {
          // `hipHostMallocDefault == 0`. Coherent / portable / write-combined flags are intentionally not set;
          // the workload is small payloads written linearly by the host and DMA-read by the GPU once.
          AMDGPUDriver::get_instance().mem_alloc_host(&pinned_metadata_scratch_, new_capacity, 0u);
        }
#endif
        pinned_metadata_scratch_capacity_ = new_capacity;
      }
      if (pinned_metadata_event_ == nullptr) {
        // `cuEventCreate` flag `0` (CU_EVENT_DEFAULT) means timing-enabled, which the driver costs us nothing
        // to set up here and lets future profilers attach without re-creating the event. `hipEventCreateWithFlags`
        // takes the same encoding.
#if defined(QD_WITH_CUDA)
        if (config_.arch == Arch::cuda) {
          CUDADriver::get_instance().event_create(&pinned_metadata_event_, 0u);
        }
#endif
#if defined(QD_WITH_AMDGPU)
        if (config_.arch == Arch::amdgpu) {
          AMDGPUDriver::get_instance().event_create(&pinned_metadata_event_, 0u);
        }
#endif
      }
      // Block until any in-flight copies from the previous launch have finished pulling from the pinned scratch
      // before we overwrite it. In steady state this is a no-op because the small DMAs finish well before the
      // host loops back here; the wait exists only to defend against an unusual interleaving where the GPU
      // queue is backlogged and the next launch enters this function before the previous launch's last copy
      // has been consumed.
      wait_pending();

      auto *pinned = static_cast<uint64_t *>(pinned_metadata_scratch_);
      pinned[0] = stride_combined_u64;
      pinned[1] = stride_float_u64;
      pinned[2] = stride_int_u64;
      std::memcpy(pinned + 3, host_offsets.data(), array_bytes);
      std::memcpy(pinned + 3 + n_stacks, host_max_sizes.data(), array_bytes);

      // Queue the metadata copies on the same stream the subsequent main-kernel dispatch will run on, so the
      // GPU stream-orders the copies before the kernel reads `adstack_max_sizes` etc. On CUDA the active
      // stream is `CUDAContext::get_instance().get_stream()` - configurable via `set_stream`, defaults to the
      // null stream - and `CUDAContext::launch` dispatches kernels on the same handle. AMDGPU has no
      // public stream-selection API: `AMDGPUContext::launch` always passes `nullptr` to `hipLaunchKernel`
      // (i.e. the default stream), so the copies match that.
#if defined(QD_WITH_CUDA)
      if (config_.arch == Arch::cuda) {
        void *active_stream = CUDAContext::get_instance().get_stream();
        CUDADriver::get_instance().memcpy_host_to_device_async(runtime_adstack_stride_field_ptr_, pinned,
                                                               sizeof(uint64_t), active_stream);
        if (runtime_adstack_stride_float_field_ptr_ != nullptr) {
          CUDADriver::get_instance().memcpy_host_to_device_async(runtime_adstack_stride_float_field_ptr_, pinned + 1,
                                                                 sizeof(uint64_t), active_stream);
        }
        if (runtime_adstack_stride_int_field_ptr_ != nullptr) {
          CUDADriver::get_instance().memcpy_host_to_device_async(runtime_adstack_stride_int_field_ptr_, pinned + 2,
                                                                 sizeof(uint64_t), active_stream);
        }
        CUDADriver::get_instance().memcpy_host_to_device_async(offsets_dev_ptr, pinned + 3, array_bytes, active_stream);
        CUDADriver::get_instance().memcpy_host_to_device_async(max_sizes_dev_ptr, pinned + 3 + n_stacks, array_bytes,
                                                               active_stream);
        CUDADriver::get_instance().event_record(pinned_metadata_event_, active_stream);
      }
#endif
#if defined(QD_WITH_AMDGPU)
      if (config_.arch == Arch::amdgpu) {
        void *active_stream = nullptr;  // AMDGPUContext::launch always uses the default stream.
        AMDGPUDriver::get_instance().memcpy_host_to_device_async(runtime_adstack_stride_field_ptr_, pinned,
                                                                 sizeof(uint64_t), active_stream);
        if (runtime_adstack_stride_float_field_ptr_ != nullptr) {
          AMDGPUDriver::get_instance().memcpy_host_to_device_async(runtime_adstack_stride_float_field_ptr_, pinned + 1,
                                                                   sizeof(uint64_t), active_stream);
        }
        if (runtime_adstack_stride_int_field_ptr_ != nullptr) {
          AMDGPUDriver::get_instance().memcpy_host_to_device_async(runtime_adstack_stride_int_field_ptr_, pinned + 2,
                                                                   sizeof(uint64_t), active_stream);
        }
        AMDGPUDriver::get_instance().memcpy_host_to_device_async(offsets_dev_ptr, pinned + 3, array_bytes,
                                                                 active_stream);
        AMDGPUDriver::get_instance().memcpy_host_to_device_async(max_sizes_dev_ptr, pinned + 3 + n_stacks, array_bytes,
                                                                 active_stream);
        AMDGPUDriver::get_instance().event_record(pinned_metadata_event_, active_stream);
      }
#endif
      pinned_metadata_event_pending_ = true;
    }
  } else {
    // GPU (CUDA / AMDGPU): encode the SizeExpr trees into device bytecode, upload, launch the sizer runtime
    // function, read back just the computed stride. The sizer kernel writes `adstack_max_sizes[]`,
    // `adstack_offsets[]`, and `adstack_per_thread_stride` directly into the runtime struct and the metadata
    // arrays above - no further host-writes to those fields are needed this launch.
    //
    // Why this architecture rather than host-eval: on CUDA / AMDGPU the ndarray data lives in GPU-private memory
    // (plain `cudaMalloc` / `hipMalloc`, not managed / unified), so the host evaluator's `ExternalTensorRead`
    // deref reads garbage. Moving the interpreter on-device keeps the pointer semantics intact - it reads the
    // data pointer out of `ctx->arg_buffer` (which the kernel will read too) and dereferences it where the
    // memory lives, with no migration / readback of the ndarray payload itself.
    std::vector<uint8_t> bytecode;
    if (program_impl_ != nullptr && program_impl_->program != nullptr) {
      bytecode = encode_adstack_size_expr_device_bytecode(ad_stack, program_impl_->program, ctx);
    } else {
      // No program attached (rare: C++-only tests that construct Program without a full runtime). Fall through
      // to compile-time bounds by emitting an empty-tree bytecode - the device interpreter sees
      // `root_node_idx == -1` for every stack and routes to `max_size_compile_time`.
      bytecode = encode_adstack_size_expr_device_bytecode(ad_stack, nullptr, ctx);
    }
    // Grow the scratch buffer if the bytecode outgrew the cached capacity. Amortised doubling keeps the
    // allocation traffic O(log max_bytecode_bytes) across a run.
    const std::size_t bytecode_bytes = bytecode.size();
    if (bytecode_bytes > adstack_sizer_bytecode_capacity_) {
      std::size_t new_cap = std::max<std::size_t>(bytecode_bytes, 2 * adstack_sizer_bytecode_capacity_);
      Device::AllocParams params{};
      params.size = new_cap;
      params.host_read = false;
      params.host_write = false;
      params.export_sharing = false;
      params.usage = AllocUsage::Storage;
      DeviceAllocation new_alloc;
      RhiResult res = llvm_device()->allocate_memory(params, &new_alloc);
      QD_ERROR_IF(res != RhiResult::success,
                  "Failed to allocate {} bytes for the adstack sizer bytecode scratch buffer (err: {})", params.size,
                  int(res));
      adstack_sizer_bytecode_alloc_ = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));
      adstack_sizer_bytecode_capacity_ = new_cap;
    }
    void *bytecode_dev_ptr = get_device_alloc_info_ptr(*adstack_sizer_bytecode_alloc_);
    copy_h2d(bytecode_dev_ptr, bytecode.data(), bytecode_bytes);

    // Invoke the device interpreter. On CUDA / AMDGPU `JITModule::call` launches this as a single-thread kernel
    // on the default stream and stream-orders it before the subsequent main-kernel dispatch, so the writes we
    // do here are visible by the time the user's kernel reads `adstack_max_sizes` etc.
    //
    // The sizer kernel dereferences `ctx->arg_buffer` on device (that's how it resolves `ExternalTensorRead` leaves
    // against ndarray pointers the caller packed into the arg buffer). AMDGPU always stages a device-side copy of
    // `RuntimeContext` because HIP has no UVA fallback and the host pointer faults with `hipErrorIllegalAddress`. CUDA
    // stages the device copy only when the driver + kernel do not expose HMM / system-allocated memory (queried via
    // `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`): CUDA UVA covers pinned / CUDA-managed memory only, not the plain
    // `std::make_unique<RuntimeContext>()` backing, so a host pointer works on HMM-capable setups but faults otherwise
    // (Turing without HMM, Windows, pre-535 Linux drivers) as `CUDA_ERROR_ILLEGAL_ADDRESS` at the next DtoH sync
    // `illegal memory access ... while calling memcpy_device_to_host`. When the caller passes `nullptr` (HMM-capable
    // CUDA) we fall back to the host pointer; the launcher gates the allocation so HMM-equipped setups pay no staging
    // cost.
    auto *const runtime_jit = get_runtime_jit_module();
    void *runtime_context_ptr_for_sizer =
        device_runtime_context_ptr != nullptr ? device_runtime_context_ptr : static_cast<void *>(&ctx->get_context());
    runtime_jit->call<void *, void *, void *>("runtime_eval_adstack_size_expr", llvm_runtime_,
                                              runtime_context_ptr_for_sizer, bytecode_dev_ptr);

    // Read back the per-kind strides published by `runtime_eval_adstack_size_expr` so we can size the float and int
    // heaps independently host-side. The combined stride is unused by the split-heap codegen but kept around for
    // legacy-kernel backward compatibility (mirrors `stride_int` in the unconditional-split layout).
    uint64_t stride_combined_readback = 0;
    uint64_t stride_float_readback = 0;
    uint64_t stride_int_readback = 0;
    copy_d2h(&stride_combined_readback, runtime_adstack_stride_field_ptr_, sizeof(uint64_t));
    if (runtime_adstack_stride_float_field_ptr_ != nullptr) {
      copy_d2h(&stride_float_readback, runtime_adstack_stride_float_field_ptr_, sizeof(uint64_t));
    }
    if (runtime_adstack_stride_int_field_ptr_ != nullptr) {
      copy_d2h(&stride_int_readback, runtime_adstack_stride_int_field_ptr_, sizeof(uint64_t));
    }
    stride = static_cast<std::size_t>(stride_combined_readback);
    stride_float_bytes = static_cast<std::size_t>(stride_float_readback);
    stride_int_bytes = static_cast<std::size_t>(stride_int_readback);
  }

  // Legacy combined heap: not allocated. The unconditional-split codegen reads `heap_float` for f32 allocas and
  // `heap_int` for i32 / u1 allocas; the legacy `adstack_heap_buffer` field is never dereferenced by freshly-compiled
  // kernels. Skipping the allocation drops ~stride_int_bytes * num_threads of unused VRAM (multiple GB on heavy
  // reverse-mode kernels on Nvidia / AMDGPU at saturating_grid_dim).
  std::size_t needed_bytes = 0;
  // Always allocate the int heap at `num_threads * stride_int_bytes` worst case. Int allocas are autodiff-emitted at
  // the offload root unconditionally (loop-counter recovery, branch flags), so every dispatched thread reaches them and
  // the eager `linear_tid * stride_int + int_offset` layout demands a row per thread.
  if (stride_int_bytes > 0) {
    const std::size_t int_bytes = stride_int_bytes * num_threads;
    if (std::getenv("QD_DEBUG_ADSTACK")) {
      std::fprintf(stderr,
                   "[adstack_heap] arch=llvm kind=I src=worst_case_num_threads num_threads=%zu stride=%zu "
                   "required_bytes=%zu (%.2f MB)\n",
                   num_threads, stride_int_bytes, int_bytes, double(int_bytes) / (1024.0 * 1024.0));
      std::fflush(stderr);
    }
    ensure_adstack_heap_int(int_bytes);
  }
  // Float heap: deferred to `ensure_per_task_float_heap_post_reducer` for tasks with a captured `bound_expr` (the
  // reducer-published count drives the sizing); for non-bound_expr tasks size at `num_threads * stride_float_bytes`
  // worst case here. The eager float path uses `linear_tid` as the row index so every dispatched thread needs backing
  // storage; only the bound_expr path can shrink to `count * stride_float_bytes`.
  if (stride_float_bytes > 0 && !ad_stack.bound_expr.has_value()) {
    const std::size_t float_bytes = stride_float_bytes * num_threads;
    if (std::getenv("QD_DEBUG_ADSTACK")) {
      std::fprintf(stderr,
                   "[adstack_heap] arch=llvm kind=F src=worst_case_num_threads_no_bound_expr num_threads=%zu "
                   "stride=%zu required_bytes=%zu (%.2f MB)\n",
                   num_threads, stride_float_bytes, float_bytes, double(float_bytes) / (1024.0 * 1024.0));
      std::fflush(stderr);
    }
    ensure_adstack_heap_float(float_bytes);
  }
  return needed_bytes;
}

}  // namespace quadrants::lang
