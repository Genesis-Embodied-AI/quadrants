#include <map>

#include "quadrants/runtime/cuda/kernel_launcher.h"
#include "quadrants/runtime/cuda/cuda_utils.h"
#include "quadrants/rhi/cuda/cuda_context.h"
#include "quadrants/rhi/cuda/cuda_driver.h"
#include "quadrants/rhi/cuda/cuda_stream_pin.h"
#include "quadrants/runtime/llvm/llvm_runtime_executor.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/program.h"

#include <vector>

namespace quadrants::lang {
namespace cuda {

namespace {

// SPIR-V's `generate_struct_for_kernel` dispatches at most 65536 threads (`advisory_total_num_threads = 65536`, see
// `quadrants/codegen/spirv/spirv_codegen.cpp`) and grid-strides over the full element list inside the kernel body. The
// CUDA / AMDGPU launcher path inherits `current_task->grid_dim = saturating_grid_dim` (~9000 blocks, ~1.15M threads on
// a 144-SM Blackwell with `query_max_block_per_sm * 2`), giving the runtime kernel ~17x more concurrent thread slots
// than SPIR-V dispatches for the same workload. Per-thread adstack heap rows scale with that, so a bound_expr-less
// reverse kernel that fits in 1.2 GB on Metal balloons to ~20 GB worst case here. `gpu_parallel_struct_for` and
// `gpu_parallel_range_for` both grid-stride (`i += grid_dim()` / `idx += block_dim() * grid_dim()`) so reducing the
// concurrent thread count is correctness-equivalent; we capped to the same 65536 advisory total to track the SPIR-V
// backend's heap footprint.
constexpr std::size_t kAdStackMaxConcurrentThreads = 65536;

// Resolve the tight thread count for a task's adstack sizing. For dynamic-bound range_for the begin / end
// i32 values live in `runtime->temporaries` on device; the launcher fetches them via a 4-byte DtoH memcpy
// each (dominated by the kernel-launch overhead that follows and only paid for kernels that actually use an
// adstack under a dynamic iteration range). Const-bound range_for and non-range_for tasks use the codegen-
// computed `static_num_threads`.
std::size_t resolve_num_threads(const AdStackSizingInfo &info, LlvmRuntimeExecutor *executor) {
  std::size_t base = info.static_num_threads;
  if (info.dynamic_gpu_range_for) {
    std::int32_t begin = info.begin_const_value;
    std::int32_t end = info.end_const_value;
    if (info.begin_offset_bytes >= 0 || info.end_offset_bytes >= 0) {
      auto *active_stream = CUDAContext::get_instance().get_stream();
      auto *temp_dev_ptr = reinterpret_cast<uint8_t *>(executor->get_runtime_temporaries_device_ptr());
      if (info.begin_offset_bytes >= 0) {
        CUDADriver::get_instance().memcpy_device_to_host_async(&begin, temp_dev_ptr + info.begin_offset_bytes,
                                                               sizeof(std::int32_t), active_stream);
      }
      if (info.end_offset_bytes >= 0) {
        CUDADriver::get_instance().memcpy_device_to_host_async(&end, temp_dev_ptr + info.end_offset_bytes,
                                                               sizeof(std::int32_t), active_stream);
      }
      CUDADriver::get_instance().stream_synchronize(active_stream);
    }
    // Clamp the logical iteration count to the launched thread count: adstack slices are indexed by
    // `linear_thread_idx()` (`block_idx * block_dim + thread_idx`), so only `static_num_threads = grid_dim * block_dim`
    // slices can ever be touched concurrently. A logical range much larger than the launch size does not need more heap
    // than `static_num_threads * per_thread_stride`; allocating the logical count would over-commit memory and trip OOM
    // paths for no gain.
    std::size_t iter = end > begin ? static_cast<std::size_t>(end - begin) : 0;
    base = std::min(iter, info.static_num_threads);
  }
  // Match the SPIR-V advisory cap on adstack-bearing kernels so the heap footprint scales with
  // `kAdStackMaxConcurrentThreads * stride` instead of `saturating_grid_dim * block_dim * stride`.
  return std::min(base, kAdStackMaxConcurrentThreads);
}

}  // namespace

void KernelLauncher::launch_offloaded_tasks(LaunchContextBuilder &ctx,
                                            JITModule *cuda_module,
                                            const std::vector<OffloadedTask> &offloaded_tasks,
                                            void *device_context_ptr) {
  auto *executor = get_runtime_executor();
  // Two gates govern the per-launch adstack publish work, both opt-in by the kernel's IR shape. Forward-only kernels
  // skip both gates and pay zero adstack overhead; reverse-mode kernels without a captured `bound_expr` skip the
  // lazy-claim block, paying the per-task `publish_adstack_metadata` only.
  //   - `any_adstack`: at least one task has an `AdStackAllocaStmt`. Gates the per-task `publish_adstack_metadata`
  //     call (sets per-thread stride for the codegen heap-base addressing).
  //   - `any_lazy_task`: at least one task has a captured `bound_expr` (the codegen routes such tasks through the
  //     lazy LCA-block atomic-rmw row claim, which reads `runtime->adstack_row_counters[task_id]` and
  //     `runtime->adstack_bound_row_capacities[task_id]`). Gates `publish_adstack_lazy_claim_buffers` and the
  //     per-task reducer dispatch + DtoH heap sizing.
  const bool any_lazy_task = std::any_of(offloaded_tasks.begin(), offloaded_tasks.end(),
                                         [](const OffloadedTask &t) { return t.ad_stack.bound_expr.has_value(); });
  if (any_lazy_task) {
    // Allocate / reset the per-kernel lazy-claim arrays once before the first task. See the matching CPU launcher
    // block for rationale; on CUDA the same memcpy_host_to_device path through the cached field pointers publishes
    // the cleared counter and UINT32_MAX-defaulted capacity arrays.
    executor->publish_adstack_lazy_claim_buffers(offloaded_tasks.size());
  }
  // Max-reducer dispatch. Runs before the per-task loop so each `publish_adstack_metadata` call sees the result map via
  // the executor's `current_max_reducer_results_` and can substitute captured `MaxOverRange`s inside its encoder. Gated
  // on whether any task has captured specs so forward-only and reverse-mode-without-recognized-MaxOverRange kernels pay
  // zero per-launch overhead (the dispatch otherwise clears the transient map, walks `offloaded_tasks`, and constructs
  // a `CudaDefaultStreamPinGuard` RAII guard on every kernel launch). Mirrors the `any_lazy_task` gate above on
  // `publish_adstack_lazy_claim_buffers`.
  const bool any_max_reducer_task =
      std::any_of(offloaded_tasks.begin(), offloaded_tasks.end(),
                  [](const OffloadedTask &t) { return !t.ad_stack.max_reducer_specs.empty(); });
  if (any_max_reducer_task) {
    executor->dispatch_max_reducers_for_tasks(offloaded_tasks, &ctx, device_context_ptr);
  }

  // Per-task adstack setup + grid-dim capping. Shared by serial and stream-parallel paths.
  auto prepare_task = [&](std::size_t task_index, const OffloadedTask &task) -> int {
    int effective_grid_dim = task.grid_dim;
    if (!task.ad_stack.allocas.empty()) {
      std::size_t n = resolve_num_threads(task.ad_stack, executor);
      // Pass the device-side `RuntimeContext` pointer through to the adstack sizer kernel. Without it the sizer
      // launches with a host pointer and the next DtoH sync trips `CUDA_ERROR_ILLEGAL_ADDRESS ...
      // memcpy_device_to_host` on GPUs whose driver + kernel cannot coherently access pageable host memory (the HMM
      // capability gated below in `launch_llvm_kernel`). `nullptr` on HMM-capable setups keeps
      // `publish_adstack_metadata`'s host-pointer fast path.
      executor->publish_adstack_metadata(task.ad_stack, n, &ctx, device_context_ptr);
      if (task.ad_stack.bound_expr.has_value()) {
        // Device-side reducer for tasks with a captured ndarray-backed `bound_expr`: a single-thread CUDA kernel
        // walks the gating ndarray, counts gate-passing threads, writes the count into
        // `runtime->adstack_bound_row_capacities[task_index]`. The codegen-emitted clamp at the float LCA-block
        // claim site reads it back. Tasks without a captured gate keep the UINT32_MAX default and the clamp stays
        // inert.
        //
        // Reducer length is the gating ndarray's full flat element count, not `n`: the lazy row-claim atomic-rmw
        // fires once per LCA execution, and `gpu_parallel_struct_for` / `gpu_parallel_range_for` grid-stride (`i +=
        // grid_dim()`) so a single dispatched thread can hit the LCA many times across one launch when the logical
        // loop span exceeds the (capped) concurrent thread count. Walking the reducer over the full ndarray length
        // keeps `bound_row_capacities[task_index]` consistent with the total claim count, which the codegen-emitted
        // bounds clamp reads. Mirrors the CPU launcher's `bound_count_length` derivation.
        std::size_t bound_count_length = n;
        if (task.ad_stack.bound_expr->field_source_kind == StaticAdStackBoundExpr::FieldSourceKind::NdArray &&
            !task.ad_stack.bound_expr->ndarray_arg_id.empty() && task.ad_stack.bound_expr->ndarray_ndim > 0 &&
            ctx.args_type != nullptr) {
          // Length = product of shape entries via `args_type`. See `runtime/cpu/kernel_launcher.cpp` for the
          // unit-stability rationale; `array_runtime_sizes` carries different units depending on the dispatch entry
          // point and would undercount by `sizeof(elem)`x for `qd.ndarray` arguments.
          int64_t flat_len = 1;
          for (int axis = 0; axis < task.ad_stack.bound_expr->ndarray_ndim; ++axis) {
            std::vector<int> indices = task.ad_stack.bound_expr->ndarray_arg_id;
            indices.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
            indices.push_back(axis);
            // get_struct_arg_host (NOT get_struct_arg): `launch_llvm_kernel` above has already swapped
            // `ctx_->arg_buffer` to a device pointer, so a plain `get_struct_arg` here would dereference device
            // memory from the host - SIGSEGV / CUDA_ERROR_ILLEGAL_ADDRESS on drivers without HMM, garbage
            // `flat_len` on HMM-capable setups. The host backing buffer (`arg_buffer_`) stays host-resident across
            // the swap and holds the same shape entries, so the host-safe variant is byte-equivalent here.
            flat_len *= int64_t(ctx.get_struct_arg_host<int32_t>(indices));
          }
          bound_count_length = static_cast<std::size_t>(std::max<int64_t>(0, flat_len));
        }
        executor->publish_per_task_bound_count_device(task_index, task.ad_stack, bound_count_length, &ctx,
                                                      device_context_ptr);
        // Size the float heap from the published gate-passing count (DtoH'd per task). Mirrors the CPU launcher's
        // post-reducer sizing call - this is what shrinks the float slab to `count * stride_float` instead of the
        // dispatched-threads worst case on sparse-grid workloads.
        executor->ensure_per_task_float_heap_post_reducer(task_index, task.ad_stack, n, &ctx);
      }
      // For adstack-bearing tasks, dispatch at most `kAdStackMaxConcurrentThreads` (matching the heap row count
      // resolved above). The runtime's grid-strided loop (`gpu_parallel_struct_for` / `gpu_parallel_range_for`,
      // `quadrants/runtime/llvm/runtime_module/runtime.cpp`) walks the full element list / range with
      // `i += grid_dim()`, so a smaller grid completes the same workload sequentially per slot. Tasks without an
      // adstack keep the codegen-emitted `task.grid_dim` (saturating_grid_dim) for max throughput.
      //
      // Floor division (not ceiling): the heap-row count `n` resolved by `resolve_num_threads` floors at
      // `kAdStackMaxConcurrentThreads`, so dispatching `cap_blocks * block_dim` threads must not exceed that count.
      // Ceiling division would over-dispatch by `block_dim - 1` threads when `block_dim` does not divide
      // `kAdStackMaxConcurrentThreads` evenly (e.g. `block_dim=192`: `ceil(65536/192)*192 = 65664`), and threads
      // with `linear_thread_idx >= 65536` would index past the heap end.
      if (task.block_dim > 0) {
        const std::size_t cap_blocks =
            std::max<std::size_t>(1u, kAdStackMaxConcurrentThreads / static_cast<std::size_t>(task.block_dim));
        effective_grid_dim =
            static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(task.grid_dim), cap_blocks));
        if (effective_grid_dim < 1) {
          effective_grid_dim = 1;
        }
      }
    }
    return effective_grid_dim;
  };

  auto *active_stream = CUDAContext::get_instance().get_stream();
  for (size_t i = 0; i < offloaded_tasks.size();) {
    const auto &task = offloaded_tasks[i];
    if (task.stream_parallel_group_id == 0) {
      int effective_grid_dim = prepare_task(i, task);
      QD_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, effective_grid_dim, task.block_dim);
      cuda_module->launch(task.name, effective_grid_dim, task.block_dim, task.dynamic_shared_array_bytes,
                          {&ctx.get_context()}, {});
      i++;
    } else {
      size_t group_start = i;
      while (i < offloaded_tasks.size() && offloaded_tasks[i].stream_parallel_group_id != 0) {
        i++;
      }

      // Run all per-task adstack setup on active_stream before recording the fence event, so that
      // publish_adstack_metadata's async H2D copies are covered by the event that pool streams wait on.
      std::vector<int> grid_dims(i - group_start);
      for (size_t j = group_start; j < i; j++) {
        grid_dims[j - group_start] = prepare_task(j, offloaded_tasks[j]);
      }

      // Record an event on the default stream so pool streams can wait for the arg_buffer upload and any per-task
      // metadata copies (memcpy_host_to_device_async on `active_stream`) without stalling the CPU.  Pool streams are
      // CU_STREAM_NON_BLOCKING and do not implicitly synchronize with the default stream.
      void *upload_event = nullptr;
      CUDADriver::get_instance().event_create(&upload_event, 0x2 /*CU_EVENT_DISABLE_TIMING*/);
      CUDADriver::get_instance().event_record(upload_event, active_stream);

      std::map<int, void *> stream_by_id;
      for (size_t j = group_start; j < i; j++) {
        int sid = offloaded_tasks[j].stream_parallel_group_id;
        if (stream_by_id.find(sid) == stream_by_id.end()) {
          stream_by_id[sid] = CUDAContext::get_instance().acquire_stream();
        }
      }

      try {
        for (auto &[sid, s] : stream_by_id) {
          CUDADriver::get_instance().stream_wait_event(s, upload_event, 0);
        }
        for (size_t j = group_start; j < i; j++) {
          const auto &t = offloaded_tasks[j];
          CUDAContext::get_instance().set_stream(stream_by_id[t.stream_parallel_group_id]);
          QD_TRACE("Launching kernel {}<<<{}, {}>>>", t.name, grid_dims[j - group_start], t.block_dim);
          cuda_module->launch(t.name, grid_dims[j - group_start], t.block_dim, t.dynamic_shared_array_bytes,
                              {&ctx.get_context()}, {});
        }

        // Join: record an event on each pool stream and make the default stream wait, so subsequent serial work on
        // active_stream orders after the parallel group without stalling the CPU.
        for (auto &[sid, s] : stream_by_id) {
          void *done = nullptr;
          CUDADriver::get_instance().event_create(&done, 0x2 /*CU_EVENT_DISABLE_TIMING*/);
          CUDADriver::get_instance().event_record(done, s);
          CUDADriver::get_instance().stream_wait_event(active_stream, done, 0);
          CUDADriver::get_instance().event_destroy(done);
        }
      } catch (...) {
        for (auto &[sid, s] : stream_by_id) {
          CUDAContext::get_instance().release_stream(s);
        }
        CUDADriver::get_instance().event_destroy(upload_event);
        CUDAContext::get_instance().set_stream(active_stream);
        throw;
      }
      for (auto &[sid, s] : stream_by_id) {
        CUDAContext::get_instance().release_stream(s);
      }
      CUDADriver::get_instance().event_destroy(upload_event);

      CUDAContext::get_instance().set_stream(active_stream);
    }
  }
}

void KernelLauncher::launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx,
                                                          JITModule *cuda_module,
                                                          const std::vector<OffloadedTask> &offloaded_tasks,
                                                          void *device_context_ptr) {
  int32_t counter_val;
  do {
    launch_offloaded_tasks(ctx, cuda_module, offloaded_tasks, device_context_ptr);
    counter_val = 0;
    auto *stream = CUDAContext::get_instance().get_stream();
    CUDADriver::get_instance().memcpy_device_to_host_async(&counter_val, ctx.graph_do_while_flag_dev_ptr,
                                                           sizeof(int32_t), stream);
    CUDADriver::get_instance().stream_synchronize(stream);
  } while (counter_val != 0);
}

void KernelLauncher::launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) {
  QD_ASSERT(handle.get_launch_id() < contexts_.size());

  if (ctx.use_graph) {
    auto &lctx = contexts_[handle.get_launch_id()];
    // Assemble per-child launch info for any nested qd.kernel-as-subgraph calls recorded on this launch context.
    // Each child was registered (via Program::register_kernel) before this launch, so its compiled artifacts live in
    // `contexts_[child_launch_id]`. Index by child_call_index so try_launch can match each launch_child task.
    std::vector<cuda::ChildLaunchInfo> child_infos;
    if (!ctx.child_launches.empty()) {
      std::size_t max_index = 0;
      for (const auto &cl : ctx.child_launches) {
        max_index = std::max(max_index, (std::size_t)cl.child_call_index + 1);
      }
      child_infos.resize(max_index);
      for (const auto &cl : ctx.child_launches) {
        QD_ASSERT(cl.child_launch_id >= 0 && (std::size_t)cl.child_launch_id < contexts_.size());
        auto &child_ctx = contexts_[cl.child_launch_id];
        cuda::ChildLaunchInfo info;
        info.child_ctx = cl.child_ctx;
        info.child_module = child_ctx.jit_module;
        info.child_parameters = child_ctx.parameters;
        info.child_tasks = &child_ctx.offloaded_tasks;
        child_infos[cl.child_call_index] = info;
      }
    }
    if (graph_manager_.try_launch(handle.get_launch_id(), ctx, lctx.jit_module, *lctx.parameters, lctx.offloaded_tasks,
                                  get_runtime_executor(), child_infos)) {
      return;
    }
  }
  graph_manager_.mark_not_used();

  // Mutable reference: per-handle persistent buffers grow on demand on first launch. See the matching comment
  // in `runtime/amdgpu/kernel_launcher.cpp` and the `Context` struct in `kernel_launcher.h` for why these have
  // to be per-handle (recursive launches from `publish_adstack_metadata` host-eval).
  auto &launcher_ctx = contexts_[handle.get_launch_id()];
  auto *executor = get_runtime_executor();
  auto *cuda_module = launcher_ctx.jit_module;
  const auto &parameters = *launcher_ctx.parameters;
  const auto &offloaded_tasks = launcher_ctx.offloaded_tasks;

  CUDAContext::get_instance().make_current();

  // |transfers| is only used for external arrays whose data is originally on
  // host. They are first transferred onto device and that device pointer is
  // stored in |device_ptrs| below. |transfers| saves its original pointer so
  // that we can copy the data back once kernel finishes. as well as the
  // temporary device allocations, which can be freed after kernel finishes. Key
  // is [arg_id, ptr_pos], where ptr_pos is TypeFactory::DATA_PTR_POS_IN_NDARRAY
  // for data_ptr and TypeFactory::GRAD_PTR_POS_IN_NDARRAY for grad_ptr. Value
  // is [host_ptr, temporary_device_alloc]. Invariant: temp_devallocs.size() !=
  // 0 <==> transfer happened.
  std::unordered_map<ArgArrayPtrKey, std::pair<void *, DeviceAllocation>, ArgArrayPtrKeyHasher> transfers;

  // |device_ptrs| stores pointers on device for all arrays args, including
  // external arrays and ndarrays, no matter whether the data is originally on
  // device or host.
  // This is the source of truth for us to look for device pointers used in CUDA
  // kernels.
  std::unordered_map<ArgArrayPtrKey, void *, ArgArrayPtrKeyHasher> device_ptrs;

  auto *active_stream = CUDAContext::get_instance().get_stream();

  // Default-stream fast path: every HtoD / DtoH / kernel-dispatch in this launcher already routes through
  // `active_stream`, so when entry `stream_ == nullptr` AND every offloaded task launches on the same `active_stream`
  // (i.e. `stream_parallel_group_id == 0`), the entire chain serialises on the legacy default stream. The
  // `stream_synchronize` barriers between phases collapse to no-ops the surrounding sync DtoH (host-blocking on
  // pageable host memory) already drains, and the `CudaDefaultStreamPinGuard` re-pins `CUDAContext::stream_` to
  // nullptr defensively across the launch in case an inner helper temporarily swaps it. Outside the fast path -
  // user-supplied stream OR any task on `stream_parallel_group_id != 0` (per-group acquired streams differ from
  // `active_stream`) - the cross-stream barriers below are load-bearing for HtoD / kernel / DtoH visibility and the
  // pin guard would silently override the user-requested stream at the kernel-launch site, so the guard stays
  // disengaged and main's sync semantics remain untouched. Closes the cross-stream-visibility break that
  // `cuMemAllocAsync`-pool buffers show on pre-Ampere hardware (Turing T4 faults at `cuLaunchKernel` with
  // `illegal-address` when consumer + producer streams differ) on the default-stream path.
  const bool all_sgid_zero = std::all_of(offloaded_tasks.begin(), offloaded_tasks.end(),
                                         [](const OffloadedTask &t) { return t.stream_parallel_group_id == 0; });
  const bool default_stream_path = (active_stream == nullptr) && all_sgid_zero;
  CudaDefaultStreamPinGuard cuda_pin(/*engage=*/default_stream_path);

  char *device_result_buffer{nullptr};
  // Launcher-global persistent `result_buffer`. See `kernel_launcher.h` for why this one is shared across handles
  // (kernel writes + synchronous host readback before any other reader runs). `arg_buffer` and `runtime_context`
  // are per-handle (in `Context`) to avoid recursive snode-reader launches clobbering the parent's args.
  const std::size_t needed_result = std::max(ctx.result_buffer_size, sizeof(uint64));
  if (needed_result > persistent_result_buffer_capacity_) {
    if (persistent_result_buffer_dev_ptr_ != nullptr) {
      CUDADriver::get_instance().mem_free_async(persistent_result_buffer_dev_ptr_, nullptr);
    }
    const std::size_t new_cap = std::max(needed_result, 2 * persistent_result_buffer_capacity_);
    CUDADriver::get_instance().malloc_async(&persistent_result_buffer_dev_ptr_, new_cap, nullptr);
    persistent_result_buffer_capacity_ = new_cap;
  }
  device_result_buffer = static_cast<char *>(persistent_result_buffer_dev_ptr_);
  ctx.get_context().runtime = executor->get_llvm_runtime();

  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &arg_id = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[arg_id];
      // Note: both numpy and PyTorch support arrays/tensors with zeros
      // in shapes, e.g., shape=(0) or shape=(100, 0, 200). This makes
      // `arr_sz` zero.
      if (arr_sz == 0) {
        continue;
      }

      ArgArrayPtrKey data_ptr_idx{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      ArgArrayPtrKey grad_ptr_idx{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
      auto data_ptr = ctx.array_ptrs[data_ptr_idx];
      auto grad_ptr = ctx.array_ptrs[grad_ptr_idx];

      if (ctx.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone) {
        // External array
        // Note: assuming both data & grad are on the same device
        if (on_cuda_device(data_ptr)) {
          // data_ptr is a raw ptr on CUDA device
          device_ptrs[data_ptr_idx] = data_ptr;
          device_ptrs[grad_ptr_idx] = grad_ptr;
        } else {
          DeviceAllocation devalloc = executor->allocate_memory_on_device(arr_sz, (uint64 *)device_result_buffer);
          device_ptrs[data_ptr_idx] = executor->get_device_alloc_info_ptr(devalloc);
          transfers[data_ptr_idx] = {data_ptr, devalloc};

          CUDADriver::get_instance().memcpy_host_to_device_async((void *)device_ptrs[data_ptr_idx], data_ptr, arr_sz,
                                                                 active_stream);
          if (grad_ptr != nullptr) {
            DeviceAllocation grad_devalloc =
                executor->allocate_memory_on_device(arr_sz, (uint64 *)device_result_buffer);
            device_ptrs[grad_ptr_idx] = executor->get_device_alloc_info_ptr(grad_devalloc);
            transfers[grad_ptr_idx] = {grad_ptr, grad_devalloc};

            CUDADriver::get_instance().memcpy_host_to_device_async((void *)device_ptrs[grad_ptr_idx], grad_ptr, arr_sz,
                                                                   active_stream);
          } else {
            device_ptrs[grad_ptr_idx] = nullptr;
          }
        }

        ctx.set_ndarray_ptrs(arg_id, (uint64)device_ptrs[data_ptr_idx], (uint64)device_ptrs[grad_ptr_idx]);
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = device_ptrs[data_ptr_idx];
        }
      } else if (arr_sz > 0) {
        // Ndarray
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        // Unwrapped raw ptr on device
        device_ptrs[data_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);

        if (grad_ptr != nullptr) {
          ptr = static_cast<DeviceAllocation *>(grad_ptr);
          device_ptrs[grad_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);
        } else {
          device_ptrs[grad_ptr_idx] = nullptr;
        }

        ctx.set_ndarray_ptrs(arg_id, (uint64)device_ptrs[data_ptr_idx], (uint64)device_ptrs[grad_ptr_idx]);
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = device_ptrs[data_ptr_idx];
        }
      }
    }
  }
  // On the default-stream fast path the post-HtoD `stream_synchronize` is redundant: HtoD goes on the null stream and
  // the subsequent `cuda_module->launch` reads `CUDAContext::stream_` (pinned to nullptr) so the kernel dispatch
  // serialises with the HtoD by null-stream ordering. Outside the fast path the barrier remains load-bearing because
  // HtoD on `active_stream` is async and per-group launches read it from a different stream.
  if (transfers.size() > 0 && !default_stream_path) {
    CUDADriver::get_instance().stream_synchronize(active_stream);
  }
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    ctx.get_context().result_buffer = (uint64 *)device_result_buffer;
  }
  // When launching on an explicit stream (active_stream != nullptr), two calls to the same kernel on different streams
  // would race on the shared per-handle arg_buffer: the second call's memcpy can overwrite the buffer while the first
  // kernel is still reading it.  Allocate a per-call ephemeral buffer in that case; the stream-ordered free below
  // ensures the memory stays live until the kernel finishes.
  const bool use_persistent_scratch = (active_stream == nullptr);
  char *device_arg_buffer = nullptr;
  void *ephemeral_arg_buffer = nullptr;
  if (ctx.arg_buffer_size > 0) {
    if (use_persistent_scratch) {
      if (ctx.arg_buffer_size > launcher_ctx.arg_buffer_capacity) {
        if (launcher_ctx.arg_buffer_dev_ptr != nullptr) {
          CUDADriver::get_instance().mem_free_async(launcher_ctx.arg_buffer_dev_ptr, nullptr);
        }
        const std::size_t new_cap = std::max<std::size_t>(ctx.arg_buffer_size, 2 * launcher_ctx.arg_buffer_capacity);
        CUDADriver::get_instance().malloc_async(&launcher_ctx.arg_buffer_dev_ptr, new_cap, nullptr);
        launcher_ctx.arg_buffer_capacity = new_cap;
      }
      device_arg_buffer = static_cast<char *>(launcher_ctx.arg_buffer_dev_ptr);
    } else {
      CUDADriver::get_instance().malloc_async(&ephemeral_arg_buffer, ctx.arg_buffer_size, active_stream);
      device_arg_buffer = static_cast<char *>(ephemeral_arg_buffer);
    }
    CUDADriver::get_instance().memcpy_host_to_device_async(device_arg_buffer, ctx.get_context().arg_buffer,
                                                           ctx.arg_buffer_size, active_stream);
    ctx.get_context().arg_buffer = device_arg_buffer;
  }

  // Stage a device-side copy of `RuntimeContext` for the adstack sizer kernel on GPUs that cannot dereference plain
  // host pointers (`malloc` / `new`) from device code. CUDA's UVA only covers pinned / managed allocations; the
  // `std::make_unique<RuntimeContext>()` backing is neither. On drivers + kernels that expose HMM / system-allocated
  // memory the sizer can read the host pointer directly and we keep the zero-staging fast path; everywhere else
  // (Turing without HMM, Windows, older Linux without an HMM-capable driver) the host-pointer read faults with
  // `CUDA_ERROR_ILLEGAL_ADDRESS` at the next DtoH sync, so we fall back to a per-launch device copy. Gate is the
  // `PAGEABLE_MEMORY_ACCESS` device attribute rather than a compute-capability threshold: HMM availability is a
  // property of the driver + kernel combo, not the GPU architecture (sm_70 / V100 with a current HMM driver is fine;
  // sm_90 / H100 on a Windows host or with a pre-535 Linux driver still needs the fallback). We additionally skip
  // the staging for forward-only launches (no task has an adstack alloca, so `publish_adstack_metadata` early-returns
  // and the staged buffer would be pure waste).
  bool needs_sizer_device_ctx = false;
  for (const auto &task : offloaded_tasks) {
    if (!task.ad_stack.allocas.empty()) {
      needs_sizer_device_ctx = true;
      break;
    }
  }
  // Gate the HMM-shortcut on whether the device walks host page tables directly (attribute 100). This is a sharper
  // predicate than `supports_pageable_memory_access` (attribute 88, which is also true on Turing-class HMM where
  // the legacy fault-and-migrate path is unsafe under multi-process pressure). On Ampere+ the device reads host
  // memory through host page tables and the staging is redundant; on Turing/Volta we always stage.
  needs_sizer_device_ctx = needs_sizer_device_ctx && !CUDAContext::get_instance().uses_host_page_tables();
  void *device_context_ptr = nullptr;
  void *ephemeral_context_ptr = nullptr;
  if (needs_sizer_device_ctx) {
    if (use_persistent_scratch) {
      if (launcher_ctx.runtime_context_dev_ptr == nullptr) {
        CUDADriver::get_instance().malloc_async(&launcher_ctx.runtime_context_dev_ptr, sizeof(RuntimeContext), nullptr);
      }
      device_context_ptr = launcher_ctx.runtime_context_dev_ptr;
    } else {
      CUDADriver::get_instance().malloc_async(&ephemeral_context_ptr, sizeof(RuntimeContext), active_stream);
      device_context_ptr = ephemeral_context_ptr;
    }
    CUDADriver::get_instance().memcpy_host_to_device_async(device_context_ptr, &ctx.get_context(),
                                                           sizeof(RuntimeContext), active_stream);
  }

  // Adstack-cache invalidation bump - see `bump_writes_for_kernel_llvm` in `program/adstack_size_expr_eval.{h,cpp}`.
  bump_writes_for_kernel_llvm(executor->get_program(), &ctx, offloaded_tasks);

  if (ctx.graph_do_while_arg_id >= 0) {
    QD_ASSERT(ctx.graph_do_while_flag_dev_ptr);
    launch_offloaded_tasks_with_do_while(ctx, cuda_module, offloaded_tasks, device_context_ptr);
  } else {
    launch_offloaded_tasks(ctx, cuda_module, offloaded_tasks, device_context_ptr);
  }
  // Persistent scratch (default-stream path): no per-launch free for the per-handle `arg_buffer` / `runtime_context`
  // or the launcher-global `result_buffer`. All live until launcher destruction; the dtor handles the final
  // `mem_free_async`.  Ephemeral buffers (explicit-stream path) are freed below.
  if (ctx.result_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_device_to_host_async(host_result_buffer, device_result_buffer,
                                                           ctx.result_buffer_size, active_stream);
  }
  // Copy data back to host. On the default-stream fast path the kernel ran on the null stream, so a sync
  // `cuMemcpyDtoH` (host-blocking, on null stream) sees the kernel's writes without an explicit cross-stream barrier
  // and host-drains the prior async `memcpy_device_to_host_async(host_result_buffer, ...)` queued on the same null
  // stream - the explicit `stream_synchronize(nullptr)` calls collapse to no-ops. Outside the fast path the barriers
  // remain load-bearing for cross-stream visibility (per-group kernel writes vs `active_stream` DtoH).
  if (transfers.size() > 0) {
    if (!default_stream_path) {
      CUDADriver::get_instance().stream_synchronize(active_stream);
    }
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      auto &idx = itr->first;
      if (default_stream_path) {
        CUDADriver::get_instance().memcpy_device_to_host(itr->second.first, (void *)device_ptrs[idx],
                                                         ctx.array_runtime_sizes[idx.arg_id]);
      } else {
        CUDADriver::get_instance().memcpy_device_to_host_async(itr->second.first, (void *)device_ptrs[idx],
                                                               ctx.array_runtime_sizes[idx.arg_id], active_stream);
      }
    }
    if (!default_stream_path) {
      CUDADriver::get_instance().stream_synchronize(active_stream);
    }
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      executor->deallocate_memory_on_device(itr->second.second);
    }
  } else if (ctx.result_buffer_size > 0 && !default_stream_path) {
    CUDADriver::get_instance().stream_synchronize(active_stream);
  }
  // Free per-call ephemeral buffers (explicit-stream path).  The free is stream-ordered: it won't execute until all
  // preceding work on active_stream (including the kernel reads) has completed.
  if (ephemeral_arg_buffer != nullptr) {
    CUDADriver::get_instance().mem_free_async(ephemeral_arg_buffer, active_stream);
  }
  if (ephemeral_context_ptr != nullptr) {
    CUDADriver::get_instance().mem_free_async(ephemeral_context_ptr, active_stream);
  }
}

KernelLauncher::~KernelLauncher() {
  // Free per-handle and launcher-global persistent scratch. `mem_free_async` queues behind any in-flight kernel
  // reads on the default stream, so the bytes stay valid until the launcher actually goes away.
  for (auto &launcher_ctx : contexts_) {
    if (launcher_ctx.arg_buffer_dev_ptr != nullptr) {
      CUDADriver::get_instance().mem_free_async(launcher_ctx.arg_buffer_dev_ptr, nullptr);
    }
    if (launcher_ctx.runtime_context_dev_ptr != nullptr) {
      CUDADriver::get_instance().mem_free_async(launcher_ctx.runtime_context_dev_ptr, nullptr);
    }
  }
  if (persistent_result_buffer_dev_ptr_ != nullptr) {
    CUDADriver::get_instance().mem_free_async(persistent_result_buffer_dev_ptr_, nullptr);
  }
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(const LLVM::CompiledKernelData &compiled) {
  QD_ASSERT(compiled.arch() == Arch::cuda);

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();
    contexts_.resize(index + 1);

    auto &ctx = contexts_[index];
    auto *executor = get_runtime_executor();

    auto data = compiled.get_internal_data().compiled_data.clone();
    auto *jit_module = executor->create_jit_module(std::move(data.module));

    // Populate ctx
    ctx.jit_module = jit_module;
    ctx.parameters = &compiled.get_internal_data().args;
    ctx.offloaded_tasks = std::move(data.tasks);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

}  // namespace cuda
}  // namespace quadrants::lang
