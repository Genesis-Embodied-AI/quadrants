#include "quadrants/runtime/program_impls/vulkan/vulkan_program.h"

#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/codegen/spirv/kernel_compiler.h"
#include "quadrants/codegen/spirv/compiled_kernel_data.h"
#include "quadrants/runtime/gfx/kernel_launcher.h"
#include "quadrants/runtime/gfx/snode_tree_manager.h"
#include "quadrants/rhi/common/host_memory_pool.h"

using namespace quadrants::lang::vulkan;

namespace quadrants::lang {

VulkanProgramImpl::VulkanProgramImpl(CompileConfig &config) : GfxProgramImpl(config) {
}

void VulkanProgramImpl::materialize_runtime(KernelProfilerBase *profiler, uint64 **result_buffer_ptr) {
  *result_buffer_ptr =
      (uint64 *)HostMemoryPool::get_instance().allocate(sizeof(uint64) * quadrants_result_buffer_entries, 8);

  VulkanDeviceCreator::Params evd_params;
  if (config->debug) {
    QD_WARN("Enabling vulkan validation layer in debug mode");
    evd_params.enable_validation_layer = true;
  }

  embedded_device_ = std::make_unique<VulkanDeviceCreator>(evd_params);

  gfx::GfxRuntime::Params params;
  params.device = embedded_device_->device();
  params.profiler = profiler;
  params.program_impl = this;
  runtime_ = std::make_unique<gfx::GfxRuntime>(std::move(params));
  snode_tree_mgr_ = std::make_unique<gfx::SNodeTreeManager>(runtime_.get());
}

void VulkanProgramImpl::enqueue_compute_op_lambda(std::function<void(Device *device, CommandList *cmdlist)> op,
                                                  const std::vector<ComputeOpImageRef> &image_refs) {
  runtime_->enqueue_compute_op_lambda(op, image_refs);
}

void VulkanProgramImpl::finalize() {
  GfxProgramImpl::finalize();
  embedded_device_.reset();
}

VulkanProgramImpl::~VulkanProgramImpl() {
  VulkanProgramImpl::finalize();
}

}  // namespace quadrants::lang
