// Stage A of the LLVM sparse-adstack-heap lazy-claim pipeline: per-launch buffer publish + bound-expression
// evaluation. See `adstack_lazy_claim/bound_eval.h` for the stage-level documentation.

#include "quadrants/runtime/llvm/adstack_lazy_claim/bound_eval.h"

#include "quadrants/runtime/llvm/llvm_runtime_executor.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/program.h"

#include <cstring>
#include <limits>
#include <vector>

#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/ir/static_adstack_bound_reducer_device.h"
#include "quadrants/ir/static_adstack_max_reducer_device.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program_impl.h"
#include "quadrants/rhi/llvm/llvm_device.h"

#include "quadrants/platform/cuda/detect_cuda.h"
#include "quadrants/rhi/cuda/cuda_context.h"
#include "quadrants/rhi/cuda/cuda_driver.h"

#include "quadrants/platform/amdgpu/detect_amdgpu.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"

#include "quadrants/rhi/cuda/cuda_stream_pin.h"

namespace quadrants::lang {

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

  // Pin the device context's `stream_` to the legacy default stream for the runtime_jit->call below; see
  // `cuda_stream_pin.h` for the cross-stream-visibility rationale. Same shape as the max-reducer dispatch.
#if defined(QD_WITH_CUDA)
  CudaDefaultStreamPinGuard cuda_default_stream_pin(config_.arch == Arch::cuda);
#endif

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

void LlvmRuntimeExecutor::dispatch_max_reducers_for_tasks(const std::vector<OffloadedTask> &tasks,
                                                          LaunchContextBuilder *ctx,
                                                          void *device_runtime_context_ptr) {
  // Seed the per-`Program` adstack-overflow registry from the cache-loaded tasks so the diagnose path can resolve any
  // cmpxchg-recorded id back to a kernel + task name. With content-hashed `registry_id` (now serialised) the dispatcher
  // and the metadata-publish substitution helper already see the correct id directly off the ad_stack, but the
  // `Program` registry stays empty across cache reloads until something registers; without this seed, an overflow on a
  // freshly cache-loaded kernel would print the generic dual-cause fallback instead of the kernel + task identity.
  // Idempotent (same hash inputs yield the same id) and cheap (one map lookup per task), and only walks tasks whose
  // `ad_stack.max_reducer_specs` is non-empty. The cast away const is safe: the OffloadedTasks live in
  // `KernelLauncher::contexts_[...].offloaded_tasks` (non-const launcher storage), and the const-ref parameter binding
  // is purely ergonomic for the read-only-after-this-point dispatch path below.
  Program *prog = (program_impl_ != nullptr) ? program_impl_->program : nullptr;
  if (prog != nullptr) {
    auto &mutable_tasks = const_cast<std::vector<OffloadedTask> &>(tasks);
    prog->adstack_cache().ensure_runtime_registry_ids_for_max_reducer(mutable_tasks);
  }
  std::vector<const AdStackSizingInfo *> ad_stacks_view;
  ad_stacks_view.reserve(tasks.size());
  for (const auto &t : tasks) {
    ad_stacks_view.push_back(&t.ad_stack);
  }
  dispatch_max_reducers_impl(static_cast<const void *>(&tasks), ad_stacks_view, ctx, device_runtime_context_ptr);
}

void LlvmRuntimeExecutor::dispatch_max_reducers_for_tasks(const std::vector<AdStackSizingInfo> &ad_stacks,
                                                          LaunchContextBuilder *ctx,
                                                          void *device_runtime_context_ptr) {
  std::vector<const AdStackSizingInfo *> ad_stacks_view;
  ad_stacks_view.reserve(ad_stacks.size());
  for (const auto &a : ad_stacks) {
    ad_stacks_view.push_back(&a);
  }
  dispatch_max_reducers_impl(static_cast<const void *>(&ad_stacks), ad_stacks_view, ctx, device_runtime_context_ptr);
}

void LlvmRuntimeExecutor::dispatch_max_reducers_impl(const void *launch_cache_key,
                                                     const std::vector<const AdStackSizingInfo *> &ad_stacks,
                                                     LaunchContextBuilder *ctx,
                                                     void *device_runtime_context_ptr) {
  using quadrants::lang::AdStackSizeExprDeviceNode;
  using quadrants::lang::EncodedMaxReducerBody;
  using quadrants::lang::LlvmAdStackMaxReducerDeviceParams;
  using quadrants::lang::MaxReducerResultMap;

  // Reset the per-launch transient to a fresh empty map so a kernel without captured specs (or an early-out below) sees
  // a non-null shared_ptr to an empty map. Consumers in `publish_adstack_metadata` dereference the pointer
  // unconditionally; keeping it non-null here removes a per-task null check.
  auto empty_results = std::make_shared<const MaxReducerResultMap>();
  current_max_reducer_results_ = empty_results;
  if (ctx == nullptr || ctx->args_type == nullptr) {
    return;
  }

  Program *prog = (program_impl_ != nullptr) ? program_impl_->program : nullptr;
  AdStackCache *cache = (prog != nullptr) ? &prog->adstack_cache() : nullptr;

  // Per-launch fast-path: when the kernel handle's recorded dependency snapshot still matches live state, the
  // dispatched results are unchanged and we can skip the per-spec cache walk + result-map rebuild entirely. See
  // `AdStackCache::try_max_reducer_launch_cache_hit` for the deps-replay contract; the matching record path runs at the
  // bottom of this function on every successful dispatch. The cache hands back a `shared_ptr` copy of its entry's map,
  // so the assignment below is a refcount bump - no map data is copied.
  if (cache != nullptr && launch_cache_key != nullptr) {
    std::shared_ptr<const MaxReducerResultMap> hit;
    if (cache->try_max_reducer_launch_cache_hit(launch_cache_key, ctx, hit)) {
      current_max_reducer_results_ = std::move(hit);
      return;
    }
  }

  // Slow path builds its result map locally; we wrap it in a `shared_ptr` once at the end so the cache entry and the
  // executor transient share the same allocation.
  MaxReducerResultMap result;

  // Pin the device context's `stream_` to the legacy default stream for the whole dispatch + DtoH read sequence.
  // See `cuda_stream_pin.h` for the cross-stream-visibility rationale.
#if defined(QD_WITH_CUDA)
  CudaDefaultStreamPinGuard cuda_default_stream_pin(config_.arch == Arch::cuda);
#endif

  // Pass 1: per-spec cache lookup. Hits drop straight into `result`; misses go to pending with back-refs to the
  // source `SerializedSizeExpr` and `StaticAdStackMaxReducerSpec`. Host-evaluation of begin / end and body bytecode
  // encoding is deferred to the per-level prepare step below, where each spec's `dependent_mor_node_idxs` have
  // already been substituted into the working tree. Mirrors the gfx variant's level-based dispatch so a captured
  // outer `MaxOverRange` whose end references a captured inner `MaxOverRange` resolves through the inner's parallel
  // dispatch instead of host-walking it (which would either trip the host evaluator's `1<<24` cap or read garbage
  // through device-resident ndarray buffers on launchers that do not host-accessibilise their data pointers).
  struct PendingMaxReducerDispatch {
    uint64_t cache_key;
    uint32_t registry_id;
    int32_t stack_id;
    int32_t mor_node_idx;
    const SerializedSizeExpr *expr;
    const StaticAdStackMaxReducerSpec *spec;
    bool dispatched{false};
    bool dropped{false};
    LlvmAdStackMaxReducerDeviceParams params;
    std::vector<uint8_t> body_bytecode;
    std::vector<AdStackCache::SizeExprReadObservation> reads;
  };
  std::vector<PendingMaxReducerDispatch> pending;
  for (std::size_t ti = 0; ti < ad_stacks.size(); ++ti) {
    const auto &ad_stack = *ad_stacks[ti];
    if (ad_stack.max_reducer_specs.empty()) {
      continue;
    }
    const uint32_t registry_id = ad_stack.registry_id;
    if (registry_id == 0) {
      continue;
    }
    for (const auto &spec : ad_stack.max_reducer_specs) {
      const uint64_t key = (static_cast<uint64_t>(registry_id) & 0xFFFFFFFFull) |
                           ((static_cast<uint64_t>(spec.stack_id) & 0xFFFFull) << 32) |
                           ((static_cast<uint64_t>(spec.mor_node_idx) & 0xFFFFull) << 48);
      if (cache != nullptr) {
        int64_t cached;
        if (cache->try_max_reducer_cache_hit(registry_id, spec.stack_id, spec.mor_node_idx, ctx, cached)) {
          result[key] = cached;
          continue;
        }
      }
      PendingMaxReducerDispatch p{};
      p.cache_key = key;
      p.registry_id = registry_id;
      p.stack_id = spec.stack_id;
      p.mor_node_idx = spec.mor_node_idx;
      p.expr = &ad_stack.size_exprs[spec.stack_id];
      p.spec = &spec;
      pending.push_back(std::move(p));
    }
  }
  if (pending.empty()) {
    // All specs hit the per-spec cache. Record the kernel-level entry so the next launch can short-circuit before the
    // per-spec walk: we still walked one hash + observation replay per spec to reach this point, and the per-launch
    // fast path above eliminates that redundant work. Wrap the result in a `shared_ptr` once so the cache entry and the
    // executor transient share the same allocation.
    auto result_ptr = std::make_shared<const MaxReducerResultMap>(std::move(result));
    if (cache != nullptr) {
      cache->record_max_reducer_launch_cache(launch_cache_key, ad_stacks, result_ptr, ctx);
    }
    current_max_reducer_results_ = std::move(result_ptr);
    return;
  }

  // Lazy-resolve the runtime-field address for `adstack_max_reducer_outputs` once per program lifetime.
  if (runtime_adstack_max_reducer_outputs_field_ptr_ == nullptr) {
    auto *const runtime_jit = get_runtime_jit_module();
    runtime_jit->call<void *>("runtime_get_adstack_max_reducer_field_ptr", llvm_runtime_);
    runtime_adstack_max_reducer_outputs_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id, result_buffer_cache_));
  }

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

  auto *const runtime_jit = get_runtime_jit_module();
  void *runtime_context_ptr_for_reducer =
      device_runtime_context_ptr != nullptr ? device_runtime_context_ptr : static_cast<void *>(&ctx->get_context());

  auto arg_buffer_offset_resolver = [&](const std::vector<int32_t> &arg_id_path) -> int32_t {
    std::vector<int> path(arg_id_path.begin(), arg_id_path.end());
    path.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
    const std::size_t byte_off = ctx->args_type->get_element_offset(path);
    if (byte_off > std::numeric_limits<int32_t>::max()) {
      return -1;
    }
    return static_cast<int32_t>(byte_off);
  };

  // Level-based dispatch: each iteration processes the specs whose `dependent_mor_node_idxs` are all in `result`
  // (cache hits + earlier rounds), substitutes those values into the working tree, host-evaluates begin / end,
  // encodes body bytecode, then dispatches the round one spec at a time (the LLVM runtime function is single-
  // threaded; batching is per-round only at the spec-prep level). Most kernels finish in one round; nested patterns
  // (e.g. outer MaxOverRange whose end contains a captured inner max-of-array) take one round per dependency depth.
  // No-progress rounds drop the remaining specs and let the per-task sizer's loud-error path absorb them.
  std::size_t dispatched_count = 0;
  std::size_t dropped_count = 0;
  while (dispatched_count + dropped_count < pending.size()) {
    std::vector<std::size_t> level_indices;
    for (std::size_t k = 0; k < pending.size(); ++k) {
      if (pending[k].dispatched || pending[k].dropped)
        continue;
      bool deps_ok = true;
      for (int32_t dep_node : pending[k].spec->dependent_mor_node_idxs) {
        const uint64_t dep_key = (static_cast<uint64_t>(pending[k].registry_id) & 0xFFFFFFFFull) |
                                 ((static_cast<uint64_t>(pending[k].stack_id) & 0xFFFFull) << 32) |
                                 ((static_cast<uint64_t>(dep_node) & 0xFFFFull) << 48);
        if (result.find(dep_key) == result.end()) {
          deps_ok = false;
          break;
        }
      }
      if (deps_ok)
        level_indices.push_back(k);
    }
    if (level_indices.empty()) {
      for (std::size_t k = 0; k < pending.size(); ++k) {
        if (!pending[k].dispatched && !pending[k].dropped) {
          pending[k].dropped = true;
          ++dropped_count;
        }
      }
      break;
    }

    // Prepare each ready spec: substitute already-resolved deps' values into the tree, host-eval begin / end, encode
    // body bytecode. Specs whose preparation fails (axis non-resolvable, length over u32 cap, body grammar reject)
    // mark `dropped` and are skipped for this round and forever.
    std::vector<std::size_t> level_dispatch;
    level_dispatch.reserve(level_indices.size());
    for (std::size_t k : level_indices) {
      const auto *spec = pending[k].spec;
      const std::size_t num_axes = spec->axis_var_ids.size();
      if (num_axes == 0 || num_axes > static_cast<std::size_t>(kAdStackMaxReducerMaxAxes)) {
        pending[k].dropped = true;
        ++dropped_count;
        continue;
      }
      const SerializedSizeExpr substituted =
          substitute_precomputed_max_over_range(*pending[k].expr, pending[k].registry_id, pending[k].stack_id, result);
      std::vector<int64_t> per_axis_begin(num_axes, 0);
      std::vector<int64_t> per_axis_length(num_axes, 0);
      bool axes_ok = true;
      uint64_t total_length = 1;
      for (std::size_t a = 0; a < num_axes; ++a) {
        const int64_t bv = evaluate_adstack_size_expr_at_node(substituted, spec->axis_begin_node_idxs[a], prog, ctx);
        const int64_t ev = evaluate_adstack_size_expr_at_node(substituted, spec->axis_end_node_idxs[a], prog, ctx);
        if (bv < 0 || ev < 0 || ev <= bv) {
          axes_ok = false;
          break;
        }
        per_axis_begin[a] = bv;
        per_axis_length[a] = ev - bv;
        total_length *= static_cast<uint64_t>(per_axis_length[a]);
        if (total_length > std::numeric_limits<uint32_t>::max()) {
          axes_ok = false;
          break;
        }
      }
      if (!axes_ok) {
        pending[k].dropped = true;
        ++dropped_count;
        continue;
      }
      EncodedMaxReducerBody encoded = encode_max_reducer_body_bytecode(
          substituted, spec->body_node_idx, spec->axis_var_ids, arg_buffer_offset_resolver, ctx, prog);
      if (encoded.body_node_count == 0) {
        pending[k].dropped = true;
        ++dropped_count;
        continue;
      }
      pending[k].params = LlvmAdStackMaxReducerDeviceParams{};
      // `output_slot` is assigned the round-local index after `level_dispatch` is finalised, just before the dispatch
      // loop below; this matches the gfx launcher's per-round output-buffer reuse pattern.
      pending[k].params.num_axes = static_cast<uint32_t>(num_axes);
      pending[k].params.body_node_count = encoded.body_node_count;
      pending[k].params.body_root_node_idx = static_cast<int32_t>(encoded.body_node_count) - 1;
      for (std::size_t a = 0; a < num_axes; ++a) {
        pending[k].params.per_axis_length[a] = static_cast<uint32_t>(per_axis_length[a]);
        pending[k].params.per_axis_begin[a] = per_axis_begin[a];
        pending[k].params.per_axis_var_id[a] = static_cast<int32_t>(a);
      }
      pending[k].body_bytecode = std::move(encoded.bytes);
      pending[k].reads = std::move(encoded.body_reads);
      level_dispatch.push_back(k);
    }
    if (level_dispatch.empty()) {
      continue;
    }

    // Lazy-grow + (re-)publish the outputs buffer for this round. Sized to `level_dispatch.size()` i64 slots so each
    // dispatched spec's `output_slot` is its position within `level_dispatch` (round-local), matching the gfx
    // launcher's per-round output-buffer reuse. Re-publishing is required if the alloc grew across rounds because the
    // runtime field stores a raw device pointer.
    const std::size_t needed_output_bytes = level_dispatch.size() * sizeof(int64_t);
    if (needed_output_bytes > adstack_max_reducer_outputs_capacity_) {
      Device::AllocParams alloc_params{};
      alloc_params.size = std::max<std::size_t>(needed_output_bytes, 2 * adstack_max_reducer_outputs_capacity_);
      alloc_params.host_read = false;
      alloc_params.host_write = true;
      alloc_params.export_sharing = false;
      alloc_params.usage = AllocUsage::Storage;
      DeviceAllocation new_alloc;
      RhiResult res = llvm_device()->allocate_memory(alloc_params, &new_alloc);
      QD_ERROR_IF(res != RhiResult::success,
                  "Failed to allocate {} bytes for adstack max reducer outputs buffer (err: {})", alloc_params.size,
                  int(res));
      adstack_max_reducer_outputs_alloc_ = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));
      adstack_max_reducer_outputs_capacity_ = alloc_params.size;
    }
    void *outputs_dev_ptr = get_device_alloc_info_ptr(*adstack_max_reducer_outputs_alloc_);
    copy_h2d(runtime_adstack_max_reducer_outputs_field_ptr_, &outputs_dev_ptr, sizeof(void *));

    // Seed every round-local output slot with INT64_MIN so the per-thread `atomic_max_i64` reductions inside
    // `runtime_eval_adstack_max_reduce` can publish the cross-product max into a known sentinel. The
    // recognizer is skipped on CPU (see the `arch_is_cpu` gate in `codegen/llvm/codegen_llvm.cpp`), so this dispatch
    // loop only runs on CUDA / AMDGPU and the parallel reducer is the only variant invoked.
    {
      std::vector<int64_t> sentinel_slots(level_dispatch.size(), static_cast<int64_t>(0x8000000000000000ll));
      copy_h2d(outputs_dev_ptr, sentinel_slots.data(), sentinel_slots.size() * sizeof(int64_t));
    }

    // Assign each ready spec's `output_slot` to its round-local index within `level_dispatch`, then h2d its params +
    // body bytecode and invoke the runtime evaluator.
    for (std::size_t i = 0; i < level_dispatch.size(); ++i) {
      const std::size_t k = level_dispatch[i];
      pending[k].params.output_slot = static_cast<uint32_t>(i);
      const std::size_t needed_params_bytes = sizeof(LlvmAdStackMaxReducerDeviceParams);
      if (needed_params_bytes > adstack_max_reducer_params_capacity_) {
        Device::AllocParams alloc_params{};
        alloc_params.size = std::max<std::size_t>(needed_params_bytes, 2 * adstack_max_reducer_params_capacity_);
        alloc_params.host_read = false;
        alloc_params.host_write = true;
        alloc_params.export_sharing = false;
        alloc_params.usage = AllocUsage::Storage;
        DeviceAllocation new_alloc;
        RhiResult res = llvm_device()->allocate_memory(alloc_params, &new_alloc);
        QD_ERROR_IF(res != RhiResult::success,
                    "Failed to allocate {} bytes for adstack max reducer params buffer (err: {})", alloc_params.size,
                    int(res));
        adstack_max_reducer_params_alloc_ = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));
        adstack_max_reducer_params_capacity_ = alloc_params.size;
      }
      void *params_dev_ptr = get_device_alloc_info_ptr(*adstack_max_reducer_params_alloc_);
      copy_h2d(params_dev_ptr, &pending[k].params, needed_params_bytes);

      const std::size_t needed_bytecode_bytes = pending[k].body_bytecode.size();
      if (needed_bytecode_bytes > adstack_max_reducer_bytecode_capacity_) {
        Device::AllocParams alloc_params{};
        alloc_params.size = std::max<std::size_t>(needed_bytecode_bytes, 2 * adstack_max_reducer_bytecode_capacity_);
        alloc_params.host_read = false;
        alloc_params.host_write = true;
        alloc_params.export_sharing = false;
        alloc_params.usage = AllocUsage::Storage;
        DeviceAllocation new_alloc;
        RhiResult res = llvm_device()->allocate_memory(alloc_params, &new_alloc);
        QD_ERROR_IF(res != RhiResult::success,
                    "Failed to allocate {} bytes for adstack max reducer bytecode buffer (err: {})", alloc_params.size,
                    int(res));
        adstack_max_reducer_bytecode_alloc_ = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));
        adstack_max_reducer_bytecode_capacity_ = alloc_params.size;
      }
      void *bytecode_dev_ptr = get_device_alloc_info_ptr(*adstack_max_reducer_bytecode_alloc_);
      copy_h2d(bytecode_dev_ptr, pending[k].body_bytecode.data(), needed_bytecode_bytes);

      // Grid-strided parallel reducer. Cap `grid_dim` at the launcher's saturating grid_dim so the dispatch shares
      // the heap-row budget the rest of the launcher uses; `block_dim` matches the codegen default. The grid-stride
      // loop handles arbitrary `total_length` so under-dispatching is harmless.
      std::uint64_t cross_product = 1;
      for (std::size_t a = 0; a < pending[k].params.num_axes; ++a) {
        cross_product *= static_cast<std::uint64_t>(pending[k].params.per_axis_length[a]);
      }
      const std::size_t block_dim = static_cast<std::size_t>(std::max(1, config_.max_block_dim));
      const std::size_t needed_threads = std::max<std::size_t>(1, static_cast<std::size_t>(cross_product));
      const std::size_t grid_dim_cap = static_cast<std::size_t>(std::max(1, config_.saturating_grid_dim));
      const std::size_t grid_dim = std::min(grid_dim_cap, (needed_threads + block_dim - 1) / block_dim);
      runtime_jit->launch<void *, void *, void *, void *>("runtime_eval_adstack_max_reduce", grid_dim, block_dim, 0,
                                                          llvm_runtime_, runtime_context_ptr_for_reducer,
                                                          params_dev_ptr, bytecode_dev_ptr);
    }

    // Read back this round's output slots. The runtime function writes int64 values at `outputs[output_slot]`; each
    // spec's `output_slot` is its round-local index within `level_dispatch`, so the d2h covers exactly the round's
    // dispatched specs.
    std::vector<int64_t> outputs_host(level_dispatch.size(), 0);
    copy_d2h(outputs_host.data(), outputs_dev_ptr, needed_output_bytes);
    for (std::size_t i = 0; i < level_dispatch.size(); ++i) {
      const std::size_t k = level_dispatch[i];
      int64_t v = outputs_host[i];
      if (v == std::numeric_limits<int64_t>::min()) {
        v = 0;
      }
      result[pending[k].cache_key] = v;
      if (cache != nullptr) {
        populate_max_reducer_body_observations(pending[k].reads, ctx, cache);
        cache->record_max_reducer_eval(pending[k].registry_id, pending[k].stack_id, pending[k].mor_node_idx, v,
                                       std::move(pending[k].reads));
      }
      pending[k].dispatched = true;
      ++dispatched_count;
    }
  }

  // Record the kernel-level launch cache so the next launch on the same kernel handle can short-circuit at the top of
  // this function. Without this, even after the per-spec cache is fully warm we still pay an O(specs)
  // hash-lookup-and-observation-replay loop per launch. The shared_ptr wrapping happens once here so the cache entry
  // and the executor transient share the same allocation.
  auto result_ptr = std::make_shared<const MaxReducerResultMap>(std::move(result));
  if (cache != nullptr) {
    cache->record_max_reducer_launch_cache(launch_cache_key, ad_stacks, result_ptr, ctx);
  }
  // Stash the result map on the executor so `publish_adstack_metadata` reads it for substitution per task.
  current_max_reducer_results_ = std::move(result_ptr);
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

}  // namespace quadrants::lang
