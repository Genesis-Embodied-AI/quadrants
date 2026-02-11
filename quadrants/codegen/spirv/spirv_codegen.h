#pragma once

#include "quadrants/util/lang_util.h"

#include "quadrants/codegen/spirv/snode_struct_compiler.h"
#include "quadrants/codegen/spirv/kernel_utils.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

namespace quadrants::lang {

class Kernel;

namespace spirv {

class KernelCodegen {
 public:
  struct Params {
    std::string ti_kernel_name;
    const Kernel *kernel{nullptr};
    const IRNode *ir_root{nullptr};
    std::vector<CompiledSNodeStructs> compiled_structs;
    Arch arch;
    DeviceCapabilityConfig caps;
    bool enable_spv_opt{true};
    const CompileConfig *compile_config{nullptr};
  };

  explicit KernelCodegen(const Params &params);

  void run(QuadrantsKernelAttributes &kernel_attribs,
           std::vector<std::vector<uint32_t>> &generated_spirv);

 private:
  Params params_;
  KernelContextAttributes ctx_attribs_;

  std::unique_ptr<spvtools::Optimizer> spirv_opt_{nullptr};
  std::unique_ptr<spvtools::SpirvTools> spirv_tools_{nullptr};
  spvtools::OptimizerOptions spirv_opt_options_;
};

}  // namespace spirv
}  // namespace quadrants::lang
