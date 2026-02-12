#pragma once
#include "quadrants/codegen/spirv/spirv_codegen.h"
#include "quadrants/codegen/spirv/snode_struct_compiler.h"
#include "quadrants/codegen/spirv/kernel_utils.h"

#include "quadrants/rhi/vulkan/vulkan_device_creator.h"
#include "quadrants/rhi/vulkan/vulkan_utils.h"
#include "quadrants/rhi/vulkan/vulkan_loader.h"
#include "quadrants/runtime/gfx/runtime.h"
#include "quadrants/runtime/gfx/snode_tree_manager.h"
#include "quadrants/rhi/vulkan/vulkan_device.h"

#include "quadrants/common/logging.h"
#include "quadrants/struct/snode_tree.h"
#include "quadrants/program/snode_expr_utils.h"
#include "quadrants/program/program_impl.h"
#include "quadrants/program/program.h"
#include "quadrants/runtime/program_impls/gfx/gfx_program.h"

#include <optional>

namespace quadrants::lang {

namespace vulkan {
class VulkanDeviceCreator;
}

class VulkanProgramImpl : public GfxProgramImpl {
 public:
  explicit VulkanProgramImpl(CompileConfig &config);
  ~VulkanProgramImpl() override;

  void materialize_runtime(KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  Device *get_compute_device() override {
    if (embedded_device_) {
      return embedded_device_->device();
    }
    return nullptr;
  }

  Device *get_graphics_device() override {
    if (embedded_device_) {
      return embedded_device_->device();
    }
    return nullptr;
  }

  void finalize() override;

  void enqueue_compute_op_lambda(
      std::function<void(Device *device, CommandList *cmdlist)> op,
      const std::vector<ComputeOpImageRef> &image_refs) override;

 private:
  std::unique_ptr<vulkan::VulkanDeviceCreator> embedded_device_{nullptr};
};
}  // namespace quadrants::lang
