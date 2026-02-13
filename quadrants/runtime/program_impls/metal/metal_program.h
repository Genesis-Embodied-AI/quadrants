#pragma once
#include "quadrants/codegen/spirv/spirv_codegen.h"
#include "quadrants/codegen/spirv/snode_struct_compiler.h"
#include "quadrants/codegen/spirv/kernel_utils.h"

#include "quadrants/rhi/metal/metal_device.h"
#include "quadrants/runtime/gfx/runtime.h"
#include "quadrants/runtime/gfx/snode_tree_manager.h"

#include "quadrants/common/logging.h"
#include "quadrants/struct/snode_tree.h"
#include "quadrants/program/snode_expr_utils.h"
#include "quadrants/program/program_impl.h"
#include "quadrants/program/program.h"
#include "quadrants/runtime/program_impls/gfx/gfx_program.h"

namespace quadrants::lang {

class MetalProgramImpl : public GfxProgramImpl {
 public:
  explicit MetalProgramImpl(CompileConfig &config);

  void materialize_runtime(KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void enqueue_compute_op_lambda(
      std::function<void(Device *device, CommandList *cmdlist)> op,
      const std::vector<ComputeOpImageRef> &image_refs) override;
};

}  // namespace quadrants::lang
