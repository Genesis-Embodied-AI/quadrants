#include "quadrants/runtime/cpu/kernel_launcher.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/program.h"
#include "quadrants/rhi/arch.h"
#include "quadrants/runtime/llvm/llvm_runtime_executor.h"

namespace quadrants::lang {
namespace cpu {

void KernelLauncher::launch_offloaded_tasks(LaunchContextBuilder &ctx,
                                            const std::vector<TaskFunc> &task_funcs,
                                            const std::vector<AdStackSizingInfo> &ad_stacks,
                                            const std::vector<std::size_t> &num_threads_per_task) {
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
  // Span every task's `publish_adstack_metadata` call below with one shared read cache.
  SizeExprLaunchScope launch_scope;
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
  for (size_t i = 0; i < task_funcs.size(); ++i) {
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
    task_funcs[i](&ctx.get_context());
    if (ctx.get_context().cpu_assert_failed)
      break;
  }
}

void KernelLauncher::launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx,
                                                          const std::vector<TaskFunc> &task_funcs,
                                                          const std::vector<AdStackSizingInfo> &ad_stacks,
                                                          const std::vector<std::size_t> &num_threads_per_task) {
  do {
    launch_offloaded_tasks(ctx, task_funcs, ad_stacks, num_threads_per_task);
  } while (ctx.get_context().cpu_assert_failed == 0 && *static_cast<int32_t *>(ctx.graph_do_while_flag_dev_ptr) != 0);
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
      if (arg_id == ctx.graph_do_while_arg_id) {
        ctx.graph_do_while_flag_dev_ptr = data_ptr;
      }
    } else if (ctx.array_runtime_sizes[arg_id] > 0) {
      uint64 host_ptr = (uint64)executor->get_device_alloc_info_ptr(*static_cast<DeviceAllocation *>(data_ptr));
      ctx.set_array_device_allocation_type(arg_id, LaunchContextBuilder::DevAllocType::kNone);
      uint64 host_ptr_grad =
          grad_ptr == nullptr ? 0
                              : (uint64)executor->get_device_alloc_info_ptr(*static_cast<DeviceAllocation *>(grad_ptr));
      ctx.set_host_accessible_ndarray_ptrs(arg_id, host_ptr, host_ptr_grad);
      if (arg_id == ctx.graph_do_while_arg_id) {
        ctx.graph_do_while_flag_dev_ptr = (void *)host_ptr;
      }
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

  if (ctx.graph_do_while_arg_id >= 0) {
    QD_ASSERT(ctx.graph_do_while_flag_dev_ptr);
    launch_offloaded_tasks_with_do_while(ctx, launcher_ctx.task_funcs, launcher_ctx.ad_stacks,
                                         launcher_ctx.num_threads_per_task);
  } else {
    launch_offloaded_tasks(ctx, launcher_ctx.task_funcs, launcher_ctx.ad_stacks, launcher_ctx.num_threads_per_task);
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
    std::vector<AdStackSizingInfo> ad_stacks;
    std::vector<std::size_t> num_threads_per_task;
    std::vector<std::vector<int>> snode_writes_per_task;
    std::vector<std::vector<int>> arr_writes_per_task;
    std::vector<std::vector<int>> arr_reads_per_task;
    task_funcs.reserve(data.tasks.size());
    ad_stacks.reserve(data.tasks.size());
    num_threads_per_task.reserve(data.tasks.size());
    snode_writes_per_task.reserve(data.tasks.size());
    arr_writes_per_task.reserve(data.tasks.size());
    arr_reads_per_task.reserve(data.tasks.size());
    for (auto &task : data.tasks) {
      auto *func_ptr = jit_module->lookup_function(task.name);
      QD_ASSERT_INFO(func_ptr, "Offloaded datum function {} not found", task.name);
      task_funcs.push_back((TaskFunc)(func_ptr));
      // CPU never takes the dynamic_gpu_range_for branch - see `AdStackSizingInfo` - so the precomputed
      // `static_num_threads` (set by `codegen_cpu.cpp` to `num_cpu_threads` for non-serial tasks and to 1
      // for serial tasks) is the exact bound, and the launcher never has to resolve anything at dispatch
      // time.
      ad_stacks.push_back(task.ad_stack);
      num_threads_per_task.push_back(task.ad_stack.static_num_threads);
      snode_writes_per_task.push_back(task.snode_writes);
      arr_writes_per_task.push_back(task.arr_writes);
      arr_reads_per_task.push_back(task.arr_reads);
    }

    // Populate ctx
    ctx.parameters = &compiled.get_internal_data().args;
    ctx.task_funcs = std::move(task_funcs);
    ctx.ad_stacks = std::move(ad_stacks);
    ctx.num_threads_per_task = std::move(num_threads_per_task);
    ctx.snode_writes_per_task = std::move(snode_writes_per_task);
    ctx.arr_writes_per_task = std::move(arr_writes_per_task);
    ctx.arr_reads_per_task = std::move(arr_reads_per_task);

    // Precompute the array-typed parameter `arg_id`s so `launch_llvm_kernel` does not have to walk the
    // full parameters list and re-check `is_array` on every invocation.
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
