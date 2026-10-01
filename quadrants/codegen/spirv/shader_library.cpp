#include "quadrants/codegen/spirv/shader_library.h"

#include "quadrants/common/logging.h"
#include "quadrants/codegen/spirv/shaders/workgroup_spv.h"
#include "spirv-tools/linker.hpp"
#include "spirv/unified1/spirv.hpp"

namespace quadrants::lang::spirv {

std::vector<uint32_t> link_workgroup_helper(const std::vector<uint32_t> &kernel) {
  std::vector<uint32_t> library(std::begin(workgroup_helper_spv), std::end(workgroup_helper_spv));
  // The helper only uses Input and Function pointers. Match the caller's module addressing model; any required
  // physical-storage capability and extension are already declared by the caller and retained by the linker.
  uint32_t addressing_model = spv::AddressingModelLogical;
  for (size_t i = 5; i < kernel.size(); i += kernel[i] >> 16) {
    if ((kernel[i] & 0xffff) == spv::OpMemoryModel) {
      addressing_model = kernel[i + 1];
      break;
    }
  }
  for (size_t i = 5; i < library.size(); i += library[i] >> 16) {
    if ((library[i] & 0xffff) == spv::OpMemoryModel) {
      library[i + 1] = addressing_model;
      break;
    }
  }
  spvtools::Context context(SPV_ENV_UNIVERSAL_1_6);
  std::string error;
  context.SetMessageConsumer([&](spv_message_level_t, const char *, const spv_position_t &, const char *message) {
    error += message;
    error += '\n';
  });
  spvtools::LinkerOptions options;
  options.SetUseHighestVersion(true);
  std::vector<uint32_t> linked;
  auto result = spvtools::Link(context, {kernel, library}, &linked, options);
  QD_ERROR_IF(result != SPV_SUCCESS, "Failed to link GLSL workgroup helper: {}", error);
  return linked;
}

}  // namespace quadrants::lang::spirv
