#include "quadrants/common/logging.h"

#include "gtest/gtest.h"
#include "spirv-tools/libspirv.hpp"
#include "spirv_msl.hpp"
#include "quadrants/codegen/spirv/spirv_ir_builder.h"
#include "quadrants/codegen/spirv/spirv_operations.h"

namespace quadrants::lang::spirv {

TEST(SpirvVersion, RejectsOlderTargets) {
  for (auto arch : {Arch::vulkan, Arch::metal}) {
    for (uint32_t version : {0u, 0x10000u, 0x10100u, 0x10200u, 0x10300u, 0x10400u}) {
      SCOPED_TRACE(version);
      DeviceCapabilityConfig caps;
      caps.set(DeviceCapability::spirv_version, version);
      IRBuilder ir(arch, &caps);
      EXPECT_THROW(ir.init_header(), std::string);
    }
  }
}

TEST(SpirvVersion, ValidatesBuffersAndGlobalInterfaces) {
  for (auto arch : {Arch::vulkan, Arch::metal}) {
    for (uint32_t version : {0x10500u, 0x10600u}) {
      SCOPED_TRACE(version);
      DeviceCapabilityConfig caps;
      caps.set(DeviceCapability::spirv_version, version);
      caps.set(DeviceCapability::spirv_has_subgroup_basic, 1);
      IRBuilder ir(arch, &caps);
      SpirvOperations ops(ir);
      ir.init_header();

      auto array = ir.buffer_argument(ir.u32_type(), 0, 0, "array");
      std::vector<std::tuple<SType, std::string, size_t>> members = {{ir.u32_type(), "value", 0}};
      auto storage = ir.buffer_struct_argument(ir.create_struct_type(members), 0, 1, "storage");
      auto uniform = ir.uniform_struct_argument(ir.create_struct_type(members), 0, 2, "uniform");
      auto shared = ir.alloca_workgroup_array(ir.get_function_array_type(ir.u32_type(), 1));
      auto storage_ptr_type = ir.get_storage_pointer_type(ir.u32_type());
      auto uniform_ptr_type = ir.get_pointer_type(ir.u32_type(), spv::StorageClassUniform);
      EXPECT_EQ(storage_ptr_type.storage_class, spv::StorageClassStorageBuffer);
      EXPECT_EQ(uniform.stype.storage_class, spv::StorageClassUniform);

      auto main = ir.new_function();
      ir.start_function(main);
      auto zero = ir.uint_immediate_number(ir.u32_type(), 0);
      auto uniform_ptr = ir.make_value(spv::OpAccessChain, uniform_ptr_type, uniform, zero);
      auto value = ir.load_variable(uniform_ptr, ir.u32_type());
      // Exercise both input-registration paths and the private globals used by the random-number generator.
      value = ops.add(value, ops.get_global_invocation_id(0));
      value = ops.add(value, ops.get_local_invocation_id(0));
      value = ops.add(value, ops.get_subgroup_invocation_id());
      value = ops.add(value, ops.rand_u32(array));
      auto shared_ptr_type = ir.get_pointer_type(ir.u32_type(), spv::StorageClassWorkgroup);
      auto shared_ptr = ir.make_value(spv::OpAccessChain, shared_ptr_type, shared, zero);
      ir.store_variable(shared_ptr, value);
      value = ir.load_variable(shared_ptr, ir.u32_type());
      ir.store_variable(ir.struct_array_access(ir.u32_type(), array, zero), value);
      auto storage_ptr = ir.make_value(spv::OpAccessChain, storage_ptr_type, storage, zero);
      ir.store_variable(storage_ptr, value);
      ir.make_inst(spv::OpReturn);
      ir.make_inst(spv::OpFunctionEnd);

      ir.commit_kernel_function(main, "main", {array, storage, uniform, shared}, {1, 1, 1});
      auto binary = ir.finalize();
      ASSERT_GT(binary.size(), 5);
      EXPECT_EQ(binary[1], version);

      auto env = version == 0x10500 ? SPV_ENV_VULKAN_1_2 : SPV_ENV_VULKAN_1_3;
      spvtools::SpirvTools tools(env);
      std::string diagnostics;
      tools.SetMessageConsumer([&](spv_message_level_t, const char *, const spv_position_t &, const char *message) {
        diagnostics += message;
        diagnostics += '\n';
      });
      EXPECT_TRUE(tools.Validate(binary)) << diagnostics;

      if (arch == Arch::metal) {
        // SPIRV-Cross hides globals omitted from a 1.4+ interface. Validate the actual Metal translation as well
        // as the binary, including on Linux hosts where the native Metal runtime is unavailable.
        spirv_cross::CompilerMSL compiler(binary);
        spirv_cross::CompilerMSL::Options options;
        options.enable_decoration_binding = true;
        options.set_msl_version(2, 1, 0);
        compiler.set_msl_options(options);
        auto msl = compiler.compile();
        EXPECT_FALSE(msl.empty());
        auto resources = compiler.get_shader_resources();
        EXPECT_EQ(resources.storage_buffers.size(), 2);
        EXPECT_EQ(resources.uniform_buffers.size(), 1);
      }
    }
  }
}

}  // namespace quadrants::lang::spirv
