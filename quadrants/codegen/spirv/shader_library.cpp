#include "quadrants/codegen/spirv/shader_library.h"

#include "quadrants/common/logging.h"
#include "quadrants/codegen/spirv/spirv_ir_builder.h"
#include "workgroup_spv.h"
#include "spirv-tools/linker.hpp"
#include "spirv/unified1/spirv.hpp"

namespace quadrants::lang::spirv {

Value IRBuilder::call_glsl_u32_helper(Value &function, const char *name, uint32_t argument_value) {
  if (function.id == 0) {
    auto parameter_type = get_pointer_type(t_uint32_, spv::StorageClassFunction);
    SType function_type;
    function_type.id = id_counter_++;
    ib_.begin(spv::OpTypeFunction).add_seq(function_type, t_uint32_, parameter_type).commit(&global_);
    function = new_value(function_type, ValueKind::kFunction);
    decorate(spv::OpDecorate, function, spv::DecorationLinkageAttributes, name, spv::LinkageTypeImport);
    ib_.begin(spv::OpFunction).add_seq(t_uint32_, function, 0, function_type).commit(&imported_functions_);
    auto parameter = new_value(parameter_type, ValueKind::kVariablePtr);
    ib_.begin(spv::OpFunctionParameter).add_seq(parameter_type, parameter).commit(&imported_functions_);
    ib_.begin(spv::OpFunctionEnd).commit(&imported_functions_);
  }
  // GLSL passes scalar function arguments through Function-storage pointers.
  auto argument = alloca_variable(t_uint32_);
  store_variable(argument, uint_immediate_number(t_uint32_, argument_value));
  return make_value(spv::OpFunctionCall, t_uint32_, function, argument);
}

std::vector<uint32_t> link_shader_helpers(const std::vector<uint32_t> &kernel) {
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
  QD_ERROR_IF(result != SPV_SUCCESS, "Failed to link GLSL shader helpers: {}", error);
  return linked;
}

}  // namespace quadrants::lang::spirv
