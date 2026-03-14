#include "quadrants/runtime/gfx/kernel_launcher.h"
#include "quadrants/codegen/spirv/compiled_kernel_data.h"

namespace quadrants::lang {
namespace gfx {

KernelLauncher::KernelLauncher(Config config) : config_(std::move(config)) {
}

void KernelLauncher::launch_kernel_with_do_while(
    KernelHandle handle,
    LaunchContextBuilder &ctx) {
  const ArgArrayPtrKey key{ctx.graph_do_while_arg_id,
                           TypeFactory::DATA_PTR_POS_IN_NDARRAY};
  auto it = ctx.array_ptrs.find(key);
  QD_ASSERT(it != ctx.array_ptrs.end());

  auto *device = config_.gfx_runtime_->get_ti_device();
  DeviceAllocation alloc = *(static_cast<DeviceAllocation *>(it->second));

  do {
    config_.gfx_runtime_->launch_kernel(handle, ctx);
    config_.gfx_runtime_->synchronize();
    void *mapped = nullptr;
    QD_ASSERT(device->map(alloc, &mapped) == RhiResult::success);
    int32_t flag_val = *static_cast<int32_t *>(mapped);
    device->unmap(alloc);
    if (flag_val == 0)
      break;
  } while (true);
}

void KernelLauncher::launch_kernel(
    const lang::CompiledKernelData &compiled_kernel_data,
    LaunchContextBuilder &ctx) {
  auto handle = register_kernel(compiled_kernel_data);

  if (ctx.graph_do_while_arg_id >= 0) {
    launch_kernel_with_do_while(handle, ctx);
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
