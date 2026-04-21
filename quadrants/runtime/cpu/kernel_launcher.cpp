#include "quadrants/runtime/cpu/kernel_launcher.h"
#include "quadrants/rhi/arch.h"
#include "quadrants/runtime/llvm/llvm_runtime_executor.h"

namespace quadrants::lang {
namespace cpu {

void KernelLauncher::launch_offloaded_tasks(LaunchContextBuilder &ctx,
                                            const std::vector<TaskFunc> &task_funcs,
                                            const std::vector<std::size_t> &ad_stack_needed_bytes) {
  auto *executor = get_runtime_executor();
  for (size_t i = 0; i < task_funcs.size(); ++i) {
    if (ad_stack_needed_bytes[i] > 0) {
      executor->ensure_adstack_heap(ad_stack_needed_bytes[i]);
    }
    task_funcs[i](&ctx.get_context());
  }
}

void KernelLauncher::launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx,
                                                          const std::vector<TaskFunc> &task_funcs,
                                                          const std::vector<std::size_t> &ad_stack_needed_bytes) {
  do {
    launch_offloaded_tasks(ctx, task_funcs, ad_stack_needed_bytes);
  } while (*static_cast<int32_t *>(ctx.graph_do_while_flag_dev_ptr) != 0);
}

void KernelLauncher::launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) {
  QD_ASSERT(handle.get_launch_id() < contexts_.size());
  auto launcher_ctx = contexts_[handle.get_launch_id()];
  auto *executor = get_runtime_executor();

  ctx.get_context().runtime = executor->get_llvm_runtime();
  // For quadrants ndarrays, context.array_ptrs saves pointer to its
  // |DeviceAllocation|, CPU backend actually want to use the raw ptr here.
  const auto &parameters = *launcher_ctx.parameters;
  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &arg_id = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      void *data_ptr = ctx.array_ptrs[{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY}];
      void *grad_ptr = ctx.array_ptrs[{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY}];

      if (ctx.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone) {
        ctx.set_ndarray_ptrs(arg_id, (uint64)data_ptr, (uint64)grad_ptr);
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = data_ptr;
        }
      } else if (ctx.array_runtime_sizes[arg_id] > 0) {
        uint64 host_ptr = (uint64)executor->get_device_alloc_info_ptr(*static_cast<DeviceAllocation *>(data_ptr));
        ctx.set_array_device_allocation_type(arg_id, LaunchContextBuilder::DevAllocType::kNone);
        uint64 host_ptr_grad =
            grad_ptr == nullptr
                ? 0
                : (uint64)executor->get_device_alloc_info_ptr(*static_cast<DeviceAllocation *>(grad_ptr));
        ctx.set_ndarray_ptrs(arg_id, host_ptr, host_ptr_grad);
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = (void *)host_ptr;
        }
      }
    }
  }
  if (ctx.graph_do_while_arg_id >= 0) {
    QD_ASSERT(ctx.graph_do_while_flag_dev_ptr);
    launch_offloaded_tasks_with_do_while(ctx, launcher_ctx.task_funcs, launcher_ctx.ad_stack_needed_bytes);
  } else {
    launch_offloaded_tasks(ctx, launcher_ctx.task_funcs, launcher_ctx.ad_stack_needed_bytes);
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
    std::vector<std::size_t> ad_stack_needed_bytes;
    task_funcs.reserve(data.tasks.size());
    ad_stack_needed_bytes.reserve(data.tasks.size());
    for (auto &task : data.tasks) {
      auto *func_ptr = jit_module->lookup_function(task.name);
      QD_ASSERT_INFO(func_ptr, "Offloaded datum function {} not found", task.name);
      task_funcs.push_back((TaskFunc)(func_ptr));
      // CPU never takes the dynamic_gpu_range_for branch - see `AdStackSizingInfo` - so the precomputed
      // `static_num_threads` (set by `codegen_cpu.cpp` to `num_cpu_threads` for non-serial tasks and to 1
      // for serial tasks) is the exact bound, and the launcher never has to resolve anything at dispatch
      // time.
      ad_stack_needed_bytes.push_back(task.ad_stack.per_thread_stride * task.ad_stack.static_num_threads);
    }

    // Populate ctx
    ctx.parameters = &compiled.get_internal_data().args;
    ctx.task_funcs = std::move(task_funcs);
    ctx.ad_stack_needed_bytes = std::move(ad_stack_needed_bytes);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

}  // namespace cpu
}  // namespace quadrants::lang
