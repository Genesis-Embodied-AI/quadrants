#include "quadrants/runtime/cpu/kernel_launcher.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/graph_do_while_driver.h"
#include "quadrants/program/program.h"
#include "quadrants/rhi/arch.h"
#include "quadrants/runtime/llvm/llvm_runtime_executor.h"

namespace quadrants::lang {
namespace cpu {

void KernelLauncher::prepare_offloaded_tasks(LaunchContextBuilder &ctx,
                                             const std::vector<TaskFunc> &task_funcs,
                                             const std::vector<AdStackSizingInfo> &ad_stacks) {
  auto *executor = get_runtime_executor();
  ctx.get_context().cpu_assert_failed = 0;
  // Two gates govern the per-launch adstack publish work, both opt-in by the kernel's IR shape. Forward-only kernels
  // skip both gates and pay zero adstack overhead; reverse-mode kernels without a captured `bound_expr` skip the
  // lazy-claim block, paying the per-task `publish_adstack_metadata` only. See the matching comment in
  // `runtime/cuda/kernel_launcher.cpp` for the role of each gate.
  const bool any_lazy_task = std::any_of(ad_stacks.begin(), ad_stacks.end(),
                                         [](const AdStackSizingInfo &a) { return a.bound_expr.has_value(); });
  if (any_lazy_task) {
    // Allocate / reset the per-kernel lazy-claim arrays once before the first task. The codegen-emitted LCA-block row
    // claim atomic-rmws into `runtime->adstack_row_counters[task_codegen_id]`; clearing the slots ensures each task
    // counts its own LCA-block-reaching threads from zero, and writing UINT32_MAX into
    // `bound_row_capacities[task_codegen_id]` keeps the codegen-emitted bounds clamp inert until the per-task host
    // reducer below tightens specific slots.
    executor->publish_adstack_lazy_claim_buffers(task_funcs.size());
  }
  // Max-reducer dispatch. Runs before the per-task `publish_adstack_metadata` loop so each call sees the dispatched
  // values via the executor's transient result map and can substitute captured `MaxOverRange`s into per-stack
  // `SerializedSizeExpr` trees inside its encoder. Gated on whether any task has captured specs so forward-only and
  // reverse-mode-without-recognized-MaxOverRange kernels pay zero per-launch overhead (the dispatch otherwise clears
  // the transient map, walks `ad_stacks`, and constructs a `Program *` / `AdStackCache *` view on every kernel launch).
  const bool any_max_reducer_task = std::any_of(
      ad_stacks.begin(), ad_stacks.end(), [](const AdStackSizingInfo &a) { return !a.max_reducer_specs.empty(); });
  if (any_max_reducer_task) {
    executor->dispatch_max_reducers_for_tasks(ad_stacks, &ctx, /*device_runtime_context_ptr=*/nullptr);
  }
}

bool KernelLauncher::run_one_offloaded_task(LaunchContextBuilder &ctx,
                                            std::size_t i,
                                            const std::vector<TaskFunc> &task_funcs,
                                            const std::vector<AdStackSizingInfo> &ad_stacks,
                                            const std::vector<std::size_t> &num_threads_per_task) {
  auto *executor = get_runtime_executor();
  {
    if (!ad_stacks[i].allocas.empty()) {
      executor->publish_adstack_metadata(ad_stacks[i], num_threads_per_task[i], &ctx);
      if (ad_stacks[i].bound_expr.has_value()) {
        // Host-side reducer for tasks with a captured ndarray-backed `bound_expr`: walks the gating ndarray, counts
        // the threads that pass the predicate, writes the count into `runtime->adstack_bound_row_capacities[i]`. The
        // codegen-emitted bounds clamp at the float LCA-block claim site reads this slot back; with the count known,
        // an over-claim (claimed_row >= count) is clamped at `count - 1` before any descendant push / load-top site
        // uses the row id.
        //
        // Length = total flat element count of the gating ndarray, derived from `ctx.args_type` shape entries. On
        // CPU `ad_stack.static_num_threads` is the worker-pool size (typically the number of CPU cores) and is
        // unrelated to the gating field's length, so it cannot be the reducer's walk bound: a gate over an N-element
        // ndarray launched on an 8-thread pool would otherwise have the reducer count gate-passing items in only
        // `[0, 8)` and clamp every later iteration's claimed row into a single alias slot. Mirrors the SPIR-V
        // launcher's `resolve_length` over `range_for_attribs->end_shape_product`.
        std::size_t bound_count_length = num_threads_per_task[i];
        using FSK = StaticAdStackBoundExpr::FieldSourceKind;
        const auto &be = *ad_stacks[i].bound_expr;
        if (be.field_source_kind == FSK::NdArray && !be.ndarray_arg_id.empty() && be.ndarray_ndim > 0 &&
            ctx.args_type != nullptr) {
          // Length = product of shape entries via `ctx.args_type->get_element_offset(...)`. `ctx.array_runtime_sizes`
          // is unsuitable because the dispatch entry point determines its units:
          // `set_arg_external_array_with_shape` stores the byte size (numpy / torch path), `set_args_ndarray` stores
          // the element count (qd.ndarray path). Walking the shape entries through `args_type` is unit-stable and
          // matches the SPIR-V launcher's `resolve_length` over `range_for_attribs->end_shape_product`.
          int64_t flat_len = 1;
          for (int axis = 0; axis < be.ndarray_ndim; ++axis) {
            std::vector<int> indices = be.ndarray_arg_id;
            indices.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
            indices.push_back(axis);
            flat_len *= int64_t(ctx.get_struct_arg<int32_t>(indices));
          }
          bound_count_length = static_cast<std::size_t>(std::max<int64_t>(0, flat_len));
        } else if (be.field_source_kind == FSK::SNode) {
          // SNode-backed gates carry the dense field's iteration count straight in the captured descriptor
          // (`snode_iter_count = leaf_desc.iter_count`, populated by the codegen-time SNode descriptor resolver).
          // Use it as the reducer walk bound so the host evaluator sees the same per-iteration count the device-side
          // reducer sees on CUDA / AMDGPU.
          bound_count_length = static_cast<std::size_t>(be.snode_iter_count);
        }
        executor->publish_per_task_bound_count_cpu(i, ad_stacks[i], bound_count_length, &ctx);
        // Size the float heap from the reducer's gate-passing count now that the capacity slot is populated. Float
        // allocas (in tasks with a captured `bound_expr`) address through `heap_float + row_id_var * stride_float +
        // float_offset`; sizing the heap at `count * stride_float` instead of the dispatched-threads worst case is
        // where the actual memory savings on sparse-grid workloads come from.
        executor->ensure_per_task_float_heap_post_reducer(i, ad_stacks[i], num_threads_per_task[i], &ctx);
      }
    }
  }
  task_funcs[i](&ctx.get_context());
  return ctx.get_context().cpu_assert_failed == 0;
}

// Run task `i` with host-branch checkpoint gating + yield observation, emulating the device-side gate and yield-check
// kernels. A skipped task (cp_id below the resume point, or any checkpoint task after a yield fired this launch) counts
// as "ran" so the caller keeps walking. Returns false only when a device-side assert fired (the caller should stop).
// `resume_point` / `yield_signal` are the host equivalents of the device scalars and persist across all iterations of a
// graph_do_while drive.
bool KernelLauncher::run_gated_task(LaunchContextBuilder &ctx,
                                    std::size_t i,
                                    const Context &launcher_ctx,
                                    int32_t &resume_point,
                                    int32_t &yield_signal) {
  const auto &checkpoint_id_per_task = launcher_ctx.checkpoint_id_per_task;
  int32_t cp_id = i < checkpoint_id_per_task.size() ? checkpoint_id_per_task[i] : -1;
  // Host-branch gating. Mirrors the CUDA-native gate kernel's `cp_id >= resume_point` check plus the yield-check
  // kernel's "skip every subsequent checkpoint task after the first yield". Tasks outside any checkpoint (cp_id < 0)
  // always run.
  if (cp_id >= 0) {
    if (cp_id < resume_point) {
      return true;
    }
    if (yield_signal != -1) {
      return true;
    }
  }
  if (!run_one_offloaded_task(ctx, i, launcher_ctx.task_funcs, launcher_ctx.ad_stacks,
                              launcher_ctx.num_threads_per_task)) {
    return false;
  }
  // Yield observation: when we just ran the last task of a yield-bearing checkpoint body, peek at the user's `yield_on`
  // flag (already a host pointer because every ndarray-backed parameter is host-resident on CPU). A non-zero value is
  // the CPU equivalent of the device yield-check firing.
  const auto &is_last_task_of_yielding_checkpoint = launcher_ctx.is_last_task_of_yielding_checkpoint;
  if (cp_id >= 0 && i < is_last_task_of_yielding_checkpoint.size() && is_last_task_of_yielding_checkpoint[i] &&
      (std::size_t)cp_id < ctx.checkpoint_yield_on_dev_ptrs.size() && ctx.checkpoint_yield_on_dev_ptrs[cp_id]) {
    int32_t *flag = static_cast<int32_t *>(ctx.checkpoint_yield_on_dev_ptrs[cp_id]);
    if (*flag != 0) {
      // First yielder wins (matches the device-side atomicCAS). Single-threaded here, so a plain `yield_signal == -1`
      // check is enough.
      if (yield_signal == -1) {
        yield_signal = cp_id;
        last_yield_cp_id_on_last_call_ = cp_id;
      }
      // The framework intentionally does NOT clear the user's flag here -- the host owns it end-to-end and must reset
      // it to `0` before each `kernel.resume(...)` call (see docs/source/user_guide/graph.md and the ``qd.checkpoint``
      // docstring). Matches the CUDA / AMDGPU / Vulkan yield-check kernels which also leave the flag alone.
    }
  }
  return true;
}

void KernelLauncher::launch_offloaded_tasks(LaunchContextBuilder &ctx, const Context &launcher_ctx) {
  prepare_offloaded_tasks(ctx, launcher_ctx.task_funcs, launcher_ctx.ad_stacks);
  // Span every task's `publish_adstack_metadata` call with one shared read cache.
  SizeExprLaunchScope launch_scope;
  // Host-side equivalents of the device-side resume_point / yield_signal scalars, reset per launch.
  int32_t resume_point = (ctx.resume_from_checkpoint < 0) ? 0 : ctx.resume_from_checkpoint;
  int32_t yield_signal = -1;
  for (std::size_t i = 0; i < launcher_ctx.task_funcs.size(); ++i) {
    if (!run_gated_task(ctx, i, launcher_ctx, resume_point, yield_signal)) {
      break;
    }
  }
}

void KernelLauncher::launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx, const Context &launcher_ctx) {
  const auto &levels = ctx.graph_do_while_levels;
  const auto &task_funcs = launcher_ctx.task_funcs;
  const auto &graph_do_while_level_per_task = launcher_ctx.graph_do_while_level_per_task;
  const bool has_top_level_task = std::any_of(graph_do_while_level_per_task.begin(),
                                              graph_do_while_level_per_task.end(), [](int l) { return l < 0; });
  if (levels.size() == 1 && !has_top_level_task) {
    // Single loop with every task inside it: preserve the historical behaviour of re-running the entire task list each
    // iteration (this also keeps the reverse-mode adstack re-setup path intact). The `!has_top_level_task` guard
    // excludes kernels that mix the loop with plain top-level for-loops, which must run exactly once -- those go
    // through the general driver below.
    //
    // `launch_offloaded_tasks` applies the checkpoint gating each pass and records a yield in
    // `last_yield_cp_id_on_last_call_`. We emulate the cond-with-yield kernel's "exit on yield" by breaking out as soon
    // as a yield fires (otherwise the loop would re-enter, skip every checkpoint because the yield set resume_point
    // past them, never decrement the counter, and spin forever). Clearing `ctx.resume_from_checkpoint` after the first
    // pass mirrors cond-with-yield's resume_point reset: a `resume(from_checkpoint=cp)` launch only skips cp_ids < cp
    // on its first iteration.
    do {
      launch_offloaded_tasks(ctx, launcher_ctx);
      if (last_yield_cp_id_on_last_call_ != -1) {
        break;
      }
      ctx.resume_from_checkpoint = -1;
    } while (ctx.get_context().cpu_assert_failed == 0 && *static_cast<int32_t *>(levels[0].flag_dev_ptr) != 0);
    return;
  }

  // Nested graph_do_while: drive the loop tree from the per-task level tags. Adstack setup is done once up front
  // (nested graph_do_while + reverse-mode adstack is not a supported combination). The checkpoint gating + yield
  // observation run inside the launch_task callback against host-side resume_point / yield_signal scalars; a yield
  // makes every enclosing level's continue_level return false, so the yield propagates out of all loops just like the
  // device cond-with-yield kernel.
  prepare_offloaded_tasks(ctx, task_funcs, launcher_ctx.ad_stacks);
  SizeExprLaunchScope launch_scope;
  int32_t resume_point = (ctx.resume_from_checkpoint < 0) ? 0 : ctx.resume_from_checkpoint;
  int32_t yield_signal = -1;
  auto launch_task = [&](int i) -> bool {
    return run_gated_task(ctx, (std::size_t)i, launcher_ctx, resume_point, yield_signal);
  };
  auto continue_level = [&](int level) -> bool {
    // Resume only skips on the first pass: once a body has run, drop back to running every checkpoint.
    resume_point = 0;
    if (yield_signal != -1) {
      return false;
    }
    return ctx.get_context().cpu_assert_failed == 0 && *static_cast<int32_t *>(levels[level].flag_dev_ptr) != 0;
  };
  run_graph_do_while((int)task_funcs.size(), graph_do_while_level_per_task, levels, launch_task, continue_level);
}

void KernelLauncher::launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) {
  QD_ASSERT(handle.get_launch_id() < contexts_.size());
  // Hold a reference to the `Context` rather than a copy. Safe because `contexts_` is a `std::deque`
  // (see `kernel_launcher.h`) - a nested `register_llvm_kernel` running inside this same launch cannot
  // relocate the entry held here.
  const auto &launcher_ctx = contexts_[handle.get_launch_id()];
  auto *executor = get_runtime_executor();

  ctx.get_context().runtime = executor->get_llvm_runtime();
  // For quadrants ndarrays, context.array_ptrs saves pointer to its |DeviceAllocation|; the CPU backend wants the raw
  // ptr here. Iterate only the precomputed array-typed `arg_id`s.
  for (int arg_id : launcher_ctx.array_arg_ids) {
    void *data_ptr = ctx.array_ptrs[{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY}];
    void *grad_ptr = ctx.array_ptrs[{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY}];

    if (ctx.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone) {
      ctx.set_host_accessible_ndarray_ptrs(arg_id, (uint64)data_ptr, (uint64)grad_ptr);
      ctx.resolve_graph_do_while_flag(arg_id, data_ptr);
    } else if (ctx.array_runtime_sizes[arg_id] > 0) {
      uint64 host_ptr = (uint64)executor->get_device_alloc_info_ptr(*static_cast<DeviceAllocation *>(data_ptr));
      ctx.set_array_device_allocation_type(arg_id, LaunchContextBuilder::DevAllocType::kNone);
      uint64 host_ptr_grad =
          grad_ptr == nullptr ? 0
                              : (uint64)executor->get_device_alloc_info_ptr(*static_cast<DeviceAllocation *>(grad_ptr));
      ctx.set_host_accessible_ndarray_ptrs(arg_id, host_ptr, host_ptr_grad);
      ctx.resolve_graph_do_while_flag(arg_id, (void *)host_ptr);
    }
  }
  // Adstack-cache invalidation bump - see `bump_writes_for_kernel_llvm` in `program/adstack/write_gen.{h,cpp}`. This
  // call is unconditional even when the launched kernel has no adstack task of its own: between two consecutive
  // forward/backward pairs (training-step boundary) the user is free to mutate any field or ndarray through a regular
  // forward-only kernel, and the loop-bound mutation limitation that bans such writes only applies WITHIN a single
  // forward/backward pair (a current design limitation, not an intended guarantee). A non-adstack kernel can
  // therefore legally write state that a later reverse-mode kernel's sizer observes; if we gated this call on
  // `kernel_has_adstack` the next reverse launch could hit a previously recorded sizer entry whose `observed_gen`
  // still matched the un-bumped current gen and replay a stale heap-size decision - overflow or silently wrong
  // gradient. The bump body itself short-circuits cheaply when the cache holds no entry that could observe this
  // kernel's writes (see `has_any_recordings()` plus the per-id gates inside `bump_writes_for_kernel_llvm`).
  bump_writes_for_kernel_llvm(executor->get_program(), &ctx, launcher_ctx.snode_writes_per_task,
                              launcher_ctx.arr_writes_per_task, launcher_ctx.arr_reads_per_task);

  // Resolve per-checkpoint `yield_on=` ndarray host pointers into `ctx.checkpoint_yield_on_dev_ptrs` so the host-branch
  // gating can read the flag after each yield-bearing checkpoint body. Mirrors
  // `cuda::GraphManager::resolve_ctx_ndarray_ptrs` but uses the already-resolved host pointers from the array_arg_ids
  // walk above instead of separately translating DeviceAllocation handles.
  ctx.checkpoint_yield_on_dev_ptrs.assign(ctx.checkpoint_yield_on_arg_ids.size(), nullptr);
  for (std::size_t cp = 0; cp < ctx.checkpoint_yield_on_arg_ids.size(); ++cp) {
    int arg_id = ctx.checkpoint_yield_on_arg_ids[cp];
    if (arg_id < 0) {
      continue;
    }
    auto data_ptr_it = ctx.array_ptrs.find({arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY});
    if (data_ptr_it != ctx.array_ptrs.end()) {
      ctx.checkpoint_yield_on_dev_ptrs[cp] = data_ptr_it->second;
    }
  }
  // Reset the cross-launch yield bookkeeping ONLY when this launch belongs to a kernel that can yield. Mirrors the CUDA
  // path, where `GraphManager::launch_cached_graph` is the only place that touches `last_yield_cp_id_on_last_call_`;
  // non-graph aux launches (e.g. the implicit `to_numpy()` LLVM kernel triggered by user code between a yielding launch
  // and the host-side `GraphStatus.yielded` check) must not clobber the value.
  //
  // The "kernel can yield" predicate is "at least one cp has a resolved `yield_on=` arg id". Kernels with no
  // checkpoints, or with checkpoints that all opted out of `yield_on=`, leave the previous value alone.
  bool kernel_can_yield = false;
  for (int arg_id : ctx.checkpoint_yield_on_arg_ids) {
    if (arg_id >= 0) {
      kernel_can_yield = true;
      break;
    }
  }
  if (kernel_can_yield) {
    last_yield_cp_id_on_last_call_ = -1;
  }
  if (ctx.has_graph_do_while()) {
    for (const auto &level : ctx.graph_do_while_levels) {
      QD_ASSERT(level.flag_dev_ptr);
    }
    launch_offloaded_tasks_with_do_while(ctx, launcher_ctx);
  } else {
    launch_offloaded_tasks(ctx, launcher_ctx);
  }
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(const LLVM::CompiledKernelData &compiled) {
  QD_ASSERT(arch_is_cpu(compiled.arch()));

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();
    contexts_.resize(index + 1);

    auto &ctx = contexts_[index];
    auto *executor = get_runtime_executor();

    auto data = compiled.get_internal_data().compiled_data.clone();
    auto *jit_module = executor->create_jit_module(std::move(data.module));

    std::vector<TaskFunc> task_funcs;
    std::vector<int32_t> checkpoint_id_per_task;
    std::vector<AdStackSizingInfo> ad_stacks;
    std::vector<std::size_t> num_threads_per_task;
    std::vector<std::vector<int>> snode_writes_per_task;
    std::vector<std::vector<int>> arr_writes_per_task;
    std::vector<std::vector<int>> arr_reads_per_task;
    std::vector<int> graph_do_while_level_per_task;
    task_funcs.reserve(data.tasks.size());
    checkpoint_id_per_task.reserve(data.tasks.size());
    ad_stacks.reserve(data.tasks.size());
    num_threads_per_task.reserve(data.tasks.size());
    snode_writes_per_task.reserve(data.tasks.size());
    arr_writes_per_task.reserve(data.tasks.size());
    arr_reads_per_task.reserve(data.tasks.size());
    graph_do_while_level_per_task.reserve(data.tasks.size());
    for (auto &task : data.tasks) {
      auto *func_ptr = jit_module->lookup_function(task.name);
      QD_ASSERT_INFO(func_ptr, "Offloaded datum function {} not found", task.name);
      task_funcs.push_back((TaskFunc)(func_ptr));
      // Slice 6: keep the per-task cp_id alongside the function pointer so the launcher's host-branch gating can decide
      // whether to skip the task on a given launch (based on `ctx.resume_from_checkpoint` and the running
      // `yield_signal`).
      checkpoint_id_per_task.push_back(task.checkpoint_id);
      // CPU never takes the dynamic_gpu_range_for branch - see `AdStackSizingInfo` - so the precomputed
      // `static_num_threads` (set by `codegen_cpu.cpp` to `num_cpu_threads` for non-serial tasks and to 1
      // for serial tasks) is the exact bound, and the launcher never has to resolve anything at dispatch
      // time.
      ad_stacks.push_back(task.ad_stack);
      num_threads_per_task.push_back(task.ad_stack.static_num_threads);
      snode_writes_per_task.push_back(task.snode_writes);
      arr_writes_per_task.push_back(task.arr_writes);
      arr_reads_per_task.push_back(task.arr_reads);
      graph_do_while_level_per_task.push_back(task.graph_do_while_level_id);
    }

    // Populate ctx
    ctx.parameters = &compiled.get_internal_data().args;
    ctx.task_funcs = std::move(task_funcs);
    ctx.checkpoint_id_per_task = std::move(checkpoint_id_per_task);
    ctx.ad_stacks = std::move(ad_stacks);
    ctx.num_threads_per_task = std::move(num_threads_per_task);
    ctx.snode_writes_per_task = std::move(snode_writes_per_task);
    ctx.arr_writes_per_task = std::move(arr_writes_per_task);
    ctx.arr_reads_per_task = std::move(arr_reads_per_task);
    ctx.graph_do_while_level_per_task = std::move(graph_do_while_level_per_task);
    // Mark, for each task, whether it is the last task in a contiguous run of the same checkpoint_id where that
    // checkpoint has a `yield_on=` parameter. The launcher reads the user's yield_on flag only at these task
    // boundaries, so a multi-task checkpoint body pays the flag-deref exactly once (not once per task).
    //
    // Note: which checkpoint_ids actually have `yield_on=` is per-LAUNCH state (depends on which kernel signature got
    // bound), not per-REGISTER state. We can't resolve that at register time. So we mark every "last task in a
    // same-cp_id run" optimistically; the launcher then gates the read on `ctx.checkpoint_yield_on_dev_ptrs[cp_id] !=
    // nullptr`, making the per-launch yield_on= set authoritative.
    ctx.is_last_task_of_yielding_checkpoint.assign(ctx.task_funcs.size(), false);
    for (std::size_t i = 0; i < ctx.task_funcs.size(); ++i) {
      int32_t cp_id = ctx.checkpoint_id_per_task[i];
      if (cp_id < 0) {
        continue;
      }
      bool is_last_in_run = (i + 1 == ctx.task_funcs.size()) || ctx.checkpoint_id_per_task[i + 1] != cp_id;
      if (is_last_in_run) {
        ctx.is_last_task_of_yielding_checkpoint[i] = true;
      }
    }

    // Precompute the array-typed parameter `arg_id`s so `launch_llvm_kernel` does not have to walk the full parameters
    // list and re-check `is_array` on every invocation.
    ctx.array_arg_ids.clear();
    for (const auto &kv : *ctx.parameters) {
      if (kv.second.is_array) {
        ctx.array_arg_ids.push_back(kv.first);
      }
    }

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

}  // namespace cpu
}  // namespace quadrants::lang
