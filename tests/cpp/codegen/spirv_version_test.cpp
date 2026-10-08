#include "quadrants/common/logging.h"

#include "gtest/gtest.h"
#include "spirv-tools/libspirv.hpp"
#include "quadrants/codegen/spirv/spirv_ir_builder.h"

namespace quadrants::lang::spirv {

TEST(SpirvVersion, RejectsOlderTargets) {
  for (auto arch : {Arch::vulkan, Arch::metal}) {
    for (uint32_t version : {0u, 0x10000u, 0x10100u, 0x10200u}) {
      // Include the current SPIR-V version in any assertion failure reported within this scope.
      SCOPED_TRACE(version);
      DeviceCapabilityConfig caps;
      caps.set(DeviceCapability::spirv_version, version);
      IRBuilder ir(arch, &caps);
      EXPECT_THROW(ir.init_header(), std::string);
    }
  }
}

TEST(SpirvVersion, ValidatesStorageAndUniformBuffers) {
  for (auto arch : {Arch::vulkan, Arch::metal}) {
    for (uint32_t version : {0x10300u, 0x10400u, 0x10500u}) {
      // Include the current SPIR-V version in any assertion failure reported within this scope.
      SCOPED_TRACE(version);
      DeviceCapabilityConfig caps;
      caps.set(DeviceCapability::spirv_version, version);
      IRBuilder ir(arch, &caps);
      ir.init_header();

      auto array = ir.buffer_argument(ir.u32_type(), 0, 0, "array");
      std::vector<std::tuple<SType, std::string, size_t>> members = {{ir.u32_type(), "value", 0}};
      auto storage = ir.buffer_struct_argument(ir.create_struct_type(members), 0, 1, "storage");
      auto uniform = ir.uniform_struct_argument(ir.create_struct_type(members), 0, 2, "uniform");
      auto storage_ptr_type = ir.get_storage_pointer_type(ir.u32_type());
      auto uniform_ptr_type = ir.get_pointer_type(ir.u32_type(), spv::StorageClassUniform);
      EXPECT_EQ(storage_ptr_type.storage_class, spv::StorageClassStorageBuffer);
      EXPECT_EQ(uniform.stype.storage_class, spv::StorageClassUniform);

      auto main = ir.new_function();
      ir.start_function(main);
      auto zero = ir.uint_immediate_number(ir.u32_type(), 0);
      auto uniform_ptr = ir.make_value(spv::OpAccessChain, uniform_ptr_type, uniform, zero);
      auto value = ir.load_variable(uniform_ptr, ir.u32_type());
      ir.store_variable(ir.struct_array_access(ir.u32_type(), array, zero), value);
      auto storage_ptr = ir.make_value(spv::OpAccessChain, storage_ptr_type, storage, zero);
      ir.store_variable(storage_ptr, value);
      ir.make_inst(spv::OpReturn);
      ir.make_inst(spv::OpFunctionEnd);

      // SPIR-V 1.3 entry-point interfaces contain only Input/Output variables; 1.4 also requires the buffers.
      std::vector<Value> entry_point_args;
      if (version >= 0x10400) {
        entry_point_args = {array, storage, uniform};
      }
      ir.commit_kernel_function(main, "main", entry_point_args, {1, 1, 1});
      auto binary = ir.finalize();
      ASSERT_GT(binary.size(), 5);
      EXPECT_EQ(binary[1], version);

      auto env = version == 0x10300 ? SPV_ENV_VULKAN_1_1
                                    : (version == 0x10400 ? SPV_ENV_VULKAN_1_1_SPIRV_1_4 : SPV_ENV_VULKAN_1_2);
      spvtools::SpirvTools tools(env);
      std::string diagnostics;
      tools.SetMessageConsumer([&](spv_message_level_t, const char *, const spv_position_t &, const char *message) {
        diagnostics += message;
        diagnostics += '\n';
      });
      EXPECT_TRUE(tools.Validate(binary)) << diagnostics;
    }
  }
}

}  // namespace quadrants::lang::spirv
