#include "quadrants/runtime/gfx/kernel_launcher.h"
#include "quadrants/codegen/spirv/compiled_kernel_data.h"

#include <algorithm>

namespace quadrants::lang {
namespace gfx {

KernelLauncher::KernelLauncher(Config config) : config_(std::move(config)) {
}

static DevicePtr resolve_gfx_flag_ptr(LaunchContextBuilder &ctx,
                                      int cond_arg_id,
                                      Device *device) {
  const ArgArrayPtrKey key{cond_arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
  auto it = ctx.array_ptrs.find(key);
  QD_ASSERT(it != ctx.array_ptrs.end());
  DeviceAllocation alloc = *(static_cast<DeviceAllocation *>(it->second));
  return alloc.get_ptr(0);
}

static int32_t read_gfx_flag(Device *device, DevicePtr dev_ptr) {
  int32_t val;
  void *host_ptr = &val;
  size_t sz = sizeof(int32_t);
  QD_ASSERT(device->readback_data(&dev_ptr, &host_ptr, &sz, 1) ==
            RhiResult::success);
  return val;
}

static void dispatch_do_while_level_gfx(
    GfxRuntime *runtime,
    Device *device,
    GfxRuntime::PreparedLaunch &prepared,
    LaunchContextBuilder &ctx,
    const std::vector<GraphDoWhileLevel> &levels,
    int level_idx) {
  const auto &lv = levels[level_idx];
  int body_start = lv.task_offset;
  int body_end = lv.task_offset + lv.total_tasks;

  DevicePtr flag_ptr = resolve_gfx_flag_ptr(ctx, lv.cond_arg_id, device);

  struct ChildInfo {
    int level_idx, task_start, task_end;
  };
  std::vector<ChildInfo> children;
  for (int i = level_idx - 1; i >= 0; --i) {
    const auto &child = levels[i];
    int cs = child.task_offset;
    int ce = cs + child.total_tasks;
    if (cs >= body_start && ce <= body_end) {
      bool gc = false;
      for (const auto &c : children)
        if (cs >= c.task_start && cs < c.task_end) {
          gc = true;
          break;
        }
      if (!gc)
        children.push_back({i, cs, ce});
    }
  }
  std::sort(children.begin(), children.end(),
            [](const ChildInfo &a, const ChildInfo &b) {
              return a.task_start < b.task_start;
            });

  int iter = 0;
  do {
    int cursor = body_start;
    for (const auto &child : children) {
      if (cursor < child.task_start)
        runtime->dispatch_task_range(prepared, cursor, child.task_start);
      dispatch_do_while_level_gfx(runtime, device, prepared, ctx, levels,
                                  child.level_idx);
      cursor = child.task_end;
    }
    if (cursor < body_end)
      runtime->dispatch_task_range(prepared, cursor, body_end);
    runtime->synchronize();

    int32_t fv = read_gfx_flag(device, flag_ptr);
    if (fv == 0)
      break;
    ++iter;
  } while (true);
}

void KernelLauncher::launch_offloaded_tasks_with_do_while(
    Handle handle,
    LaunchContextBuilder &ctx) {
  auto *runtime = config_.gfx_runtime_;
  auto *device = runtime->get_ti_device();

  if (!ctx.graph_do_while_levels.empty()) {
    auto prepared = runtime->prepare_launch(handle, ctx);
    int outermost = static_cast<int>(ctx.graph_do_while_levels.size()) - 1;
    dispatch_do_while_level_gfx(runtime, device, prepared, ctx,
                                ctx.graph_do_while_levels, outermost);
    runtime->finalize_launch(prepared, ctx);
  } else {
    DevicePtr flag_ptr =
        resolve_gfx_flag_ptr(ctx, ctx.graph_do_while_arg_id, device);
    do {
      runtime->launch_kernel(handle, ctx);
      runtime->synchronize();
    } while (read_gfx_flag(device, flag_ptr) != 0);
  }
}

void KernelLauncher::launch_kernel(
    const lang::CompiledKernelData &compiled_kernel_data,
    LaunchContextBuilder &ctx) {
  auto handle = register_kernel(compiled_kernel_data);

  if (ctx.graph_do_while_arg_id >= 0 || !ctx.graph_do_while_levels.empty()) {
    launch_offloaded_tasks_with_do_while(handle, ctx);
  } else {
    config_.gfx_runtime_->launch_kernel(handle, ctx);
  }
}

KernelLauncher::Handle KernelLauncher::register_kernel(
    const lang::CompiledKernelData &compiled_kernel_data) {
  if (!compiled_kernel_data.get_handle()) {
    const auto *spirv_compiled =
        dynamic_cast<const spirv::CompiledKernelData *>(&compiled_kernel_data);
    const auto &spirv_data = spirv_compiled->get_internal_data();
    gfx::GfxRuntime::RegisterParams params;
    params.kernel_attribs = spirv_data.metadata.kernel_attribs;
    params.task_spirv_source_codes = spirv_data.src.spirv_src;
    params.num_snode_trees = spirv_data.metadata.num_snode_trees;
    auto h = config_.gfx_runtime_->register_quadrants_kernel(std::move(params));
    compiled_kernel_data.set_handle(h);
  }
  return *compiled_kernel_data.get_handle();
}

}  // namespace gfx
}  // namespace quadrants::lang
