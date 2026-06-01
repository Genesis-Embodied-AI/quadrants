// Stage B of the LLVM sparse-adstack-heap lazy-claim pipeline: per-launch metadata publish. See
// `adstack_lazy_claim/metadata_publish.h` for the stage-level documentation.

#include "quadrants/runtime/llvm/adstack_lazy_claim/metadata_publish.h"

#include "quadrants/runtime/llvm/llvm_runtime_executor.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/program.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "quadrants/ir/adstack_size_expr_device.h"
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

#include "quadrants/rhi/cuda/cuda_stream_pin.h"

namespace quadrants::lang {

std::size_t LlvmRuntimeExecutor::publish_adstack_metadata(const AdStackSizingInfo &ad_stack,
                                                          std::size_t num_threads,
                                                          LaunchContextBuilder *ctx,
                                                          void *device_runtime_context_ptr) {
  const auto n_stacks = ad_stack.allocas.size();
  if (n_stacks == 0 || num_threads == 0) {
    return 0;
  }

  // Pin the device context's `stream_` to the legacy default stream for the per-task sizer dispatch +
  // memcpy chain below; see `cuda_stream_pin.h` for the cross-stream-visibility rationale. The device sizer
  // (`runtime_eval_adstack_size_expr`) reads ndarray shape / data through `ctx->arg_buffer` just like the
  // max-reducer dispatch.
#if defined(QD_WITH_CUDA)
  CudaDefaultStreamPinGuard cuda_default_stream_pin(config_.arch == Arch::cuda);
#endif
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
  // The pointed-to scratch allocations are stable across launches (only `grow_to` swaps them). Skip the per-launch
  // h2d that publishes the pointer values whenever they have not changed since the last call. On HIP / CUDA each
  // skipped pointer-publish is one queue round-trip the launcher would otherwise pay; on a typical reverse-mode
  // sweep this fires thousands of times.
  if (offsets_dev_ptr != adstack_offsets_dev_ptr_published_) {
    copy_h2d(runtime_adstack_offsets_field_ptr_, &offsets_dev_ptr, sizeof(void *));
    adstack_offsets_dev_ptr_published_ = offsets_dev_ptr;
  }
  if (max_sizes_dev_ptr != adstack_max_sizes_dev_ptr_published_) {
    copy_h2d(runtime_adstack_max_sizes_field_ptr_, &max_sizes_dev_ptr, sizeof(void *));
    adstack_max_sizes_dev_ptr_published_ = max_sizes_dev_ptr;
  }

  std::size_t stride = 0;
  const bool is_gpu_llvm = (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu);

  // Shared GPU async publish helper: pack `[stride_combined, stride_float, stride_int, offsets[n_stacks],
  // max_sizes[n_stacks]]` into the pinned-host scratch (grow on demand, double-amortised), then issue 5 async H2Ds
  // on the active stream and record the completion event. Used by both the host-eval branch (CUDA / AMDGPU
  // resolvable size_exprs) and the on-device-sizer cache-hit branch. The driver's H2D DMA reads from the pinned
  // bytes at execution time, so a `wait_pending()` at the top of the next call defends against an unusual
  // interleaving where the GPU queue is backlogged and the next launch enters before the previous launch's last
  // copy has been consumed. Only callable when `is_gpu_llvm` is true.
  auto publish_metadata_pinned_async = [&](const uint64_t *offsets_src, const uint64_t *max_sizes_src,
                                           uint64_t stride_combined_u64, uint64_t stride_float_u64,
                                           uint64_t stride_int_u64) {
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
      const std::size_t new_capacity = std::max<std::size_t>(total_bytes, 2 * pinned_metadata_scratch_capacity_);
#if defined(QD_WITH_CUDA)
      if (config_.arch == Arch::cuda) {
        CUDADriver::get_instance().mem_alloc_host(&pinned_metadata_scratch_, new_capacity);
      }
#endif
#if defined(QD_WITH_AMDGPU)
      if (config_.arch == Arch::amdgpu) {
        // `hipHostMallocDefault == 0`. Coherent / portable / write-combined flags are intentionally not set; the
        // workload is small payloads written linearly by the host and DMA-read by the GPU once.
        AMDGPUDriver::get_instance().mem_alloc_host(&pinned_metadata_scratch_, new_capacity, 0u);
      }
#endif
      pinned_metadata_scratch_capacity_ = new_capacity;
    }
    if (pinned_metadata_event_ == nullptr) {
      // `cuEventCreate` flag `0` (CU_EVENT_DEFAULT) means timing-enabled, which the driver costs us nothing to set
      // up here and lets future profilers attach without re-creating the event. `hipEventCreateWithFlags` takes
      // the same encoding.
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
    wait_pending();
    auto *pinned = static_cast<uint64_t *>(pinned_metadata_scratch_);
    pinned[0] = stride_combined_u64;
    pinned[1] = stride_float_u64;
    pinned[2] = stride_int_u64;
    std::memcpy(pinned + 3, offsets_src, array_bytes);
    std::memcpy(pinned + 3 + n_stacks, max_sizes_src, array_bytes);
    // Queue the metadata copies on the stream the subsequent main-kernel dispatch will run on, so the GPU
    // stream-orders the copies before the kernel reads `adstack_max_sizes` etc. CUDA: `CUDAContext::get_stream()`
    // (configurable via `set_stream`, defaults to the null stream); AMDGPU: always the default stream because
    // `AMDGPUContext::launch` passes `nullptr` to `hipLaunchKernel`.
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
      void *active_stream = nullptr;
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
      AMDGPUDriver::get_instance().memcpy_host_to_device_async(offsets_dev_ptr, pinned + 3, array_bytes, active_stream);
      AMDGPUDriver::get_instance().memcpy_host_to_device_async(max_sizes_dev_ptr, pinned + 3 + n_stacks, array_bytes,
                                                               active_stream);
      AMDGPUDriver::get_instance().event_record(pinned_metadata_event_, active_stream);
    }
#endif
    pinned_metadata_event_pending_ = true;
  };

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
    // Span the per-stack `evaluate_adstack_size_expr` calls below with one shared read cache.
    SizeExprLaunchScope launch_scope;
    // Snapshot the dispatched-results map for this kernel before the per-stack walk. The body of any captured
    // `MaxOverRange` may host-resolve a `FieldLoad` leaf via `read_field_with_launch_cache`, which dispatches a
    // snode-reader kernel that reenters `dispatch_max_reducers_for_tasks` and rewrites `current_max_reducer_results_`
    // to point at that recursive call's result map. Reading the live executor field per stack would let that recursive
    // overwrite turn `stack_id == 0` 's substitution branch into `stack_id == 1` 's empty-map fallback - whose host
    // walk then trips the per-task sizer's `1<<24` cap on out-of-grammar shapes that the recognizer DID capture. Pin
    // the snapshot via a `shared_ptr` copy (refcount bump only, no map data copied) so the substitution loop stays
    // self-consistent and the cache-entry's allocation stays alive even if the executor's transient gets repointed
    // mid-walk.
    const auto local_max_reducer_results = current_max_reducer_results_;
    std::vector<uint64_t> host_max_sizes(n_stacks);
    for (std::size_t i = 0; i < n_stacks; ++i) {
      const SerializedSizeExpr *expr = (i < ad_stack.size_exprs.size()) ? &ad_stack.size_exprs[i] : nullptr;
      int64_t v = -1;
      if (expr != nullptr && !expr->nodes.empty() && prog != nullptr) {
        // Substitute any captured `MaxOverRange` whose result the max-reducer dispatched into a `Const` before the host
        // evaluator walks the tree. Mirrors `eval_per_task_metadata_on_host` on the SPIR-V side. The empty-results fast
        // path passes the live `expr` pointer directly so `size_expr_cache_` (keyed by `SerializedSizeExpr *`) stays
        // warm across launches; the non-empty branch builds a stack-local substituted tree and routes through
        // `evaluate_adstack_size_expr_no_cache` so the transient pointer never aliases unrelated cache entries. The
        // `shared_ptr` is initialised to a non-null empty-map sentinel by `dispatch_max_reducers_impl`, so the deref is
        // always safe.
        if (!local_max_reducer_results || local_max_reducer_results->empty()) {
          v = evaluate_adstack_size_expr(*expr, prog, ctx);
        } else {
          const SerializedSizeExpr substituted = substitute_precomputed_max_over_range(
              *expr, ad_stack.registry_id, static_cast<int32_t>(i), *local_max_reducer_results);
          v = evaluate_adstack_size_expr_no_cache(substituted, prog, ctx);
        }
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
      publish_metadata_pinned_async(host_offsets.data(), host_max_sizes.data(), stride_combined_u64, stride_float_u64,
                                    stride_int_u64);
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
    //
    // Per-task metadata cache fast path: the sizer kernel's output (offsets / max_sizes / strides) is a
    // deterministic function of (a) the per-task `AdStackSizingInfo *` (compile-time bytecode shape, stable
    // for the kernel's lifetime), (b) every SNode value a `FieldLoad` leaf reads, and (c) every ndarray
    // value an `ExternalTensorRead` leaf reads. Each launcher (cpu / cuda / amdgpu) bumps
    // `Program::snode_write_gen_` / `ndarray_data_gen_` for everything this kernel may mutate before
    // calling here, so the per-source generation snapshots stored alongside the cached payload catch any
    // input change between launches and force a fresh sizer dispatch when needed. On hit, the cached
    // offsets / max_sizes / strides are republished into the runtime struct via the same `copy_h2d` paths
    // the host-eval branch above uses, and the entire bytecode-encode + h2d + sizer-kernel launch +
    // 3x DtoH-stride pipeline is skipped. The cost of the sizer dispatch + DtoH stalls is small per
    // launch on CUDA / AMDGPU, but a long sequence of reverse-mode launches over the same kernel
    // pays it once per launch; the cache amortises that to once per generation-bump.
    Program *prog = (program_impl_ != nullptr) ? program_impl_->program : nullptr;
    bool llvm_metadata_cache_hit = false;
    if (prog != nullptr) {
      AdStackCache::LlvmPerTaskAdStackCacheEntry entry;
      if (prog->adstack_cache().try_llvm_per_task_ad_stack_cache_hit(static_cast<const void *>(&ad_stack), ctx,
                                                                     entry)) {
        QD_ASSERT(entry.offsets.size() == n_stacks && entry.max_sizes.size() == n_stacks);
        // Publish the cached payload through the pinned-host async pipeline shared with the host-eval
        // branch above: one pinned-scratch pack + five `memcpy_host_to_device_async` issued on the same
        // stream the main kernel will dispatch on, ordered behind the previous launch's
        // `pinned_metadata_event_pending_` wait. Packing the same `[stride_combined, stride_float,
        // stride_int, offsets[n_stacks], max_sizes[n_stacks]]` shape keeps both branches' DMA pattern
        // identical and removes the per-launch sync round-trips a `copy_h2d` would otherwise impose; on
        // CPU `copy_h2d` is `memcpy` already so we keep the direct path there.
        if (!is_gpu_llvm) {
          copy_h2d(offsets_dev_ptr, entry.offsets.data(), n_stacks * sizeof(uint64_t));
          copy_h2d(max_sizes_dev_ptr, entry.max_sizes.data(), n_stacks * sizeof(uint64_t));
          copy_h2d(runtime_adstack_stride_field_ptr_, &entry.stride_combined, sizeof(uint64_t));
          if (runtime_adstack_stride_float_field_ptr_ != nullptr) {
            copy_h2d(runtime_adstack_stride_float_field_ptr_, &entry.stride_float, sizeof(uint64_t));
          }
          if (runtime_adstack_stride_int_field_ptr_ != nullptr) {
            copy_h2d(runtime_adstack_stride_int_field_ptr_, &entry.stride_int, sizeof(uint64_t));
          }
        } else {
          publish_metadata_pinned_async(entry.offsets.data(), entry.max_sizes.data(), entry.stride_combined,
                                        entry.stride_float, entry.stride_int);
        }
        stride = static_cast<std::size_t>(entry.stride_combined);
        stride_float_bytes = static_cast<std::size_t>(entry.stride_float);
        stride_int_bytes = static_cast<std::size_t>(entry.stride_int);
        llvm_metadata_cache_hit = true;
      }
    }
    if (!llvm_metadata_cache_hit) {
      std::vector<uint8_t> bytecode;
      // The encoder reads the result map by `const &`. `current_max_reducer_results_` is a `shared_ptr<const map>`
      // initialised to a non-null empty-map sentinel by `dispatch_max_reducers_impl`, so the deref is safe; defend
      // anyway against the path where `dispatch_max_reducers_impl` was never invoked (forward-only kernels) by falling
      // back to a stack-local empty map.
      static const MaxReducerResultMap kEmptyResults{};
      const MaxReducerResultMap &results_view =
          current_max_reducer_results_ ? *current_max_reducer_results_ : kEmptyResults;
      if (program_impl_ != nullptr && program_impl_->program != nullptr) {
        bytecode = encode_adstack_size_expr_device_bytecode(ad_stack, program_impl_->program, ctx, results_view);
      } else {
        // No program attached (rare: C++-only tests that construct Program without a full runtime). Fall through
        // to compile-time bounds by emitting an empty-tree bytecode - the device interpreter sees
        // `root_node_idx == -1` for every stack and routes to `max_size_compile_time`.
        bytecode = encode_adstack_size_expr_device_bytecode(ad_stack, nullptr, ctx, results_view);
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
      // `RuntimeContext` because HIP has no UVA fallback and the host pointer faults with `hipErrorIllegalAddress`.
      // CUDA stages the device copy only when the driver + kernel do not expose HMM / system-allocated memory (queried
      // via `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`): CUDA UVA covers pinned / CUDA-managed memory only, not the
      // plain `std::make_unique<RuntimeContext>()` backing, so a host pointer works on HMM-capable setups but faults
      // otherwise (Turing without HMM, Windows, pre-535 Linux drivers) as `CUDA_ERROR_ILLEGAL_ADDRESS` at the next DtoH
      // sync `illegal memory access ... while calling memcpy_device_to_host`. When the caller passes `nullptr`
      // (HMM-capable CUDA) we fall back to the host pointer; the launcher gates the allocation so HMM-equipped setups
      // pay no staging cost.
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

      // Record the cache entry so the next launch on this kernel can skip the sizer pipeline. We also
      // need to read back the offsets / max_sizes arrays the sizer wrote to the device buffers - the
      // cache hit path above republishes them, so we must store host copies here. n_stacks is small
      // (a few dozen at most for any reasonable kernel) so the extra DtoH cost is negligible
      // compared to the dispatch + sizer-kernel launch we are about to amortise away.
      if (prog != nullptr) {
        std::vector<uint64_t> offsets_readback(n_stacks);
        std::vector<uint64_t> max_sizes_readback(n_stacks);
        copy_d2h(offsets_readback.data(), offsets_dev_ptr, n_stacks * sizeof(uint64_t));
        copy_d2h(max_sizes_readback.data(), max_sizes_dev_ptr, n_stacks * sizeof(uint64_t));
        // Walk size_exprs structurally to gather the dependency keys (snode_ids referenced via
        // FieldLoad, arg_ids referenced via ExternalTensorShape / ExternalTensorRead). Pure tree
        // inspection - no live value reads, no nested kernel launches. Mirrors the SPIR-V analogue.
        std::unordered_set<int> snode_ids;
        std::unordered_set<int> arg_ids;
        for (const auto &expr : ad_stack.size_exprs) {
          for (const auto &node : expr.nodes) {
            switch (static_cast<SizeExpr::Kind>(node.kind)) {
              case SizeExpr::Kind::FieldLoad:
                if (node.snode_id >= 0)
                  snode_ids.insert(node.snode_id);
                break;
              case SizeExpr::Kind::ExternalTensorShape:
              case SizeExpr::Kind::ExternalTensorRead:
                if (!node.arg_id_path.empty())
                  arg_ids.insert(node.arg_id_path.front());
                break;
              default:
                break;
            }
          }
        }
        std::vector<std::pair<int, uint64_t>> snode_gens;
        snode_gens.reserve(snode_ids.size());
        for (int snode_id : snode_ids) {
          snode_gens.emplace_back(snode_id, prog->adstack_cache().snode_write_gen(snode_id));
        }
        std::vector<std::tuple<int, void *, uint64_t>> arg_gens;
        arg_gens.reserve(arg_ids.size());
        for (int arg_id : arg_ids) {
          ArgArrayPtrKey data_key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
          auto ap_it = ctx->array_ptrs.find(data_key);
          void *devalloc = (ap_it == ctx->array_ptrs.end()) ? nullptr : ap_it->second;
          arg_gens.emplace_back(arg_id, devalloc, prog->adstack_cache().ndarray_data_gen(devalloc));
        }
        prog->adstack_cache().record_llvm_per_task_ad_stack(
            static_cast<const void *>(&ad_stack), std::move(offsets_readback), std::move(max_sizes_readback),
            stride_combined_readback, stride_float_readback, stride_int_readback, std::move(snode_gens),
            std::move(arg_gens));
      }
    }  // end if (!llvm_metadata_cache_hit)
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
  last_published_stride_float_bytes_ = stride_float_bytes;
  return needed_bytes;
}

}  // namespace quadrants::lang
