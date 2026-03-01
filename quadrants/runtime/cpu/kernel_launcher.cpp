#include "quadrants/runtime/cpu/kernel_launcher.h"
#include "quadrants/rhi/arch.h"

namespace quadrants::lang {
namespace cpu {

void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
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
      void *data_ptr =
          ctx.array_ptrs[{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY}];
      void *grad_ptr =
          ctx.array_ptrs[{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY}];

      if (ctx.device_allocation_type[arg_id] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        ctx.set_ndarray_ptrs(arg_id, (uint64)data_ptr, (uint64)grad_ptr);
        if (arg_id == ctx.graph_while_arg_id) {
          ctx.graph_while_flag_dev_ptr = data_ptr;
        }
      } else if (ctx.array_runtime_sizes[arg_id] > 0) {
        uint64 host_ptr = (uint64)executor->get_device_alloc_info_ptr(
            *static_cast<DeviceAllocation *>(data_ptr));
        ctx.set_array_device_allocation_type(
            arg_id, LaunchContextBuilder::DevAllocType::kNone);
        uint64 host_ptr_grad =
            grad_ptr == nullptr
                ? 0
                : (uint64)executor->get_device_alloc_info_ptr(
                      *static_cast<DeviceAllocation *>(grad_ptr));
        ctx.set_ndarray_ptrs(arg_id, host_ptr, host_ptr_grad);
        if (arg_id == ctx.graph_while_arg_id) {
          ctx.graph_while_flag_dev_ptr = (void *)host_ptr;
        }
      }
    }
  }
  do {
    for (auto task : launcher_ctx.task_funcs) {
      task(&ctx.get_context());
    }
  } while (ctx.graph_while_arg_id >= 0 &&
           ctx.graph_while_flag_dev_ptr &&
           *static_cast<int32_t *>(ctx.graph_while_flag_dev_ptr) != 0);
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(
    const LLVM::CompiledKernelData &compiled) {
  QD_ASSERT(arch_is_cpu(compiled.arch()));

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();
    contexts_.resize(index + 1);

    auto &ctx = contexts_[index];
    auto *executor = get_runtime_executor();

    auto data = compiled.get_internal_data().compiled_data.clone();
    auto *jit_module = executor->create_jit_module(std::move(data.module));

    // Construct task_funcs
    using TaskFunc = int32 (*)(void *);
    std::vector<TaskFunc> task_funcs;
    task_funcs.reserve(data.tasks.size());
    for (auto &task : data.tasks) {
      auto *func_ptr = jit_module->lookup_function(task.name);
      QD_ASSERT_INFO(func_ptr, "Offloaded datum function {} not found",
                     task.name);
      task_funcs.push_back((TaskFunc)(func_ptr));
    }

    // Populate ctx
    ctx.parameters = &compiled.get_internal_data().args;
    ctx.task_funcs = std::move(task_funcs);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

}  // namespace cpu
}  // namespace quadrants::lang
