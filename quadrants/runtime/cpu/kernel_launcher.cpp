#include "quadrants/runtime/cpu/kernel_launcher.h"
#include "quadrants/rhi/arch.h"

#include <algorithm>

namespace quadrants::lang {
namespace cpu {

void KernelLauncher::launch_offloaded_tasks(
    LaunchContextBuilder &ctx,
    const std::vector<TaskFunc> &task_funcs) {
  for (auto task : task_funcs) {
    task(&ctx.get_context());
  }
}

using TaskFunc_ = int32 (*)(void *);

static void launch_task_range(LaunchContextBuilder &ctx,
                              const std::vector<TaskFunc_> &task_funcs,
                              int start,
                              int end) {
  for (int t = start; t < end; ++t) {
    task_funcs[t](&ctx.get_context());
  }
}

static void dispatch_do_while_level_cpu(
    LaunchContextBuilder &ctx,
    const std::vector<TaskFunc_> &task_funcs,
    const std::vector<GraphDoWhileLevel> &levels,
    int level_idx) {
  const auto &lv = levels[level_idx];
  int body_start = lv.task_offset;
  int body_end = lv.task_offset + lv.total_tasks;

  struct ChildInfo {
    int level_idx;
    int task_start;
    int task_end;
  };
  std::vector<ChildInfo> children;
  for (int i = level_idx - 1; i >= 0; --i) {
    const auto &child = levels[i];
    int cs = child.task_offset;
    int ce = child.task_offset + child.total_tasks;
    if (cs >= body_start && ce <= body_end) {
      bool is_grandchild = false;
      for (const auto &c : children) {
        if (cs >= c.task_start && cs < c.task_end) {
          is_grandchild = true;
          break;
        }
      }
      if (!is_grandchild) {
        children.push_back({i, cs, ce});
      }
    }
  }
  std::sort(children.begin(), children.end(),
            [](const ChildInfo &a, const ChildInfo &b) {
              return a.task_start < b.task_start;
            });

  do {
    int cursor = body_start;
    for (const auto &child : children) {
      if (cursor < child.task_start) {
        launch_task_range(ctx, task_funcs, cursor, child.task_start);
      }
      dispatch_do_while_level_cpu(ctx, task_funcs, levels, child.level_idx);
      cursor = child.task_end;
    }
    if (cursor < body_end) {
      launch_task_range(ctx, task_funcs, cursor, body_end);
    }
  } while (*static_cast<int32_t *>(lv.flag_dev_ptr) != 0);
}

void KernelLauncher::launch_offloaded_tasks_with_do_while(
    LaunchContextBuilder &ctx,
    const std::vector<TaskFunc> &task_funcs) {
  if (ctx.graph_do_while_levels.empty()) {
    do {
      launch_offloaded_tasks(ctx, task_funcs);
    } while (*static_cast<int32_t *>(ctx.graph_do_while_flag_dev_ptr) != 0);
  } else {
    int outermost = static_cast<int>(ctx.graph_do_while_levels.size()) - 1;
    dispatch_do_while_level_cpu(ctx, task_funcs, ctx.graph_do_while_levels,
                                outermost);
  }
}

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
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = data_ptr;
        }
        for (auto &lv : ctx.graph_do_while_levels) {
          if (arg_id == lv.cond_arg_id) {
            lv.flag_dev_ptr = data_ptr;
          }
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
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = (void *)host_ptr;
        }
        for (auto &lv : ctx.graph_do_while_levels) {
          if (arg_id == lv.cond_arg_id) {
            lv.flag_dev_ptr = (void *)host_ptr;
          }
        }
      }
    }
  }
  if (ctx.graph_do_while_arg_id >= 0) {
    QD_ASSERT(ctx.graph_do_while_flag_dev_ptr);
    launch_offloaded_tasks_with_do_while(ctx, launcher_ctx.task_funcs);
  } else {
    launch_offloaded_tasks(ctx, launcher_ctx.task_funcs);
  }
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
