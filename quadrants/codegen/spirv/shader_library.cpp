#include "quadrants/codegen/spirv/shader_library.h"

#include <cstdint>

#include "quadrants/common/logging.h"
#include "quadrants/codegen/spirv/spirv_ir_builder.h"
#include "workgroup_spv.h"
#include "spirv-tools/linker.hpp"
#include "spirv/unified1/spirv.hpp"

namespace quadrants::lang::spirv {

Value IRBuilder::call_glsl_helper(Value &function,
                                 const char *name,
                                 const SType &return_type,
                                 const std::vector<Value> &arguments) {
  if (function.id == 0) {
    std::vector<uint32_t> parameter_types;
    for (const auto &argument : arguments) {
      parameter_types.push_back(get_pointer_type(argument.stype, spv::StorageClassFunction).id);
    }
    SType function_type;
    function_type.id = id_counter_++;
    ib_.begin(spv::OpTypeFunction).add_seq(function_type, return_type, parameter_types).commit(&global_);
    function = new_value(function_type, ValueKind::kFunction);
    decorate(spv::OpDecorate, function, spv::DecorationLinkageAttributes, name, spv::LinkageTypeImport);
    ib_.begin(spv::OpFunction).add_seq(return_type, function, 0, function_type).commit(&imported_functions_);
    for (auto parameter_type : parameter_types) {
      ib_.begin(spv::OpFunctionParameter).add_seq(parameter_type, id_counter_++).commit(&imported_functions_);
    }
    ib_.begin(spv::OpFunctionEnd).commit(&imported_functions_);
  }
  // GLSL passes scalar function arguments through Function-storage pointers.
  std::vector<uint32_t> argument_pointers;
  for (const auto &argument : arguments) {
    auto pointer = alloca_variable(argument.stype);
    store_variable(pointer, argument);
    argument_pointers.push_back(pointer.id);
  }
  return make_value(spv::OpFunctionCall, return_type, function, argument_pointers);
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
