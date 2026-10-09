#include "quadrants/common/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "quadrants/codegen/spirv/spirv_operations.h"
#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/optimizer.hpp"

namespace quadrants::lang::spirv {

TEST(WorkgroupShader, LinksAndValidatesAcrossTargets) {
  for (Arch arch : {Arch::vulkan, Arch::metal}) {
    for (int version : {0x10000, 0x10300, 0x10600}) {
      for (bool physical : {false, true}) {
        if (physical && (arch == Arch::metal || version < 0x10300)) {
          continue;
        }
        SCOPED_TRACE(fmt::format("arch={} version={} physical={}", arch_name(arch), version, physical));
        DeviceCapabilityConfig caps;
        caps.set(DeviceCapability::spirv_version, version);
        caps.set(DeviceCapability::spirv_has_physical_storage_buffer, physical);
        IRBuilder ir(arch, &caps);
        SpirvOperations ops_(ir);
        ir.init_header();
        Value output = ir.buffer_argument(/* value_type= */ ir.u32_type(), /* descriptor_set= */ 0,
                                          /* binding= */ 0, /* name= */ "result");
        Value main = ir.new_function();
        ir.start_function(main);
        for (uint32_t dim : {0u, 1u, 2u, 0u}) {
          Value index = ops_.get_work_group_id(dim);
          Value destination =
              ir.struct_array_access(ir.u32_type(), output, ir.uint_immediate_number(ir.u32_type(), dim));
          ir.store_variable(destination, index);
        }
        ir.make_inst(spv::OpReturn);
        ir.make_inst(spv::OpFunctionEnd);
        ir.commit_kernel_function(/* func= */ main, /* name= */ "main", /* args= */ {output},
                                  /* local_size= */ {1, 1, 1});
        std::vector<uint32_t> spirv_module = ir.finalize();
        EXPECT_EQ(spirv_module[1], version);
        spv_target_env environment = version >= 0x10600   ? SPV_ENV_VULKAN_1_3
                                     : version >= 0x10300 ? SPV_ENV_VULKAN_1_1
                                                          : SPV_ENV_VULKAN_1_0;
        spvtools::SpirvTools tools(environment);
        std::string diagnostics;
        spvtools::MessageConsumer append_to_diagnostics = [&](spv_message_level_t, const char *, const spv_position_t &,
                                                              const char *message) {
          diagnostics += message;
          diagnostics += '\n';
        };
        tools.SetMessageConsumer(append_to_diagnostics);
        ASSERT_TRUE(tools.Validate(spirv_module)) << diagnostics;
        std::string disassembly;
        ASSERT_TRUE(tools.Disassemble(spirv_module, &disassembly));
        EXPECT_THAT(disassembly, ::testing::HasSubstr("BuiltIn WorkgroupId"));
        // Verify that linking resolved the helper and removed its import/export metadata.
        EXPECT_THAT(disassembly, ::testing::Not(::testing::HasSubstr("LinkageAttributes")));
        spvtools::Optimizer optimizer(environment);
        optimizer.SetMessageConsumer(append_to_diagnostics);
        optimizer.RegisterPerformancePasses();
        std::vector<uint32_t> optimized;
        ASSERT_TRUE(optimizer.Run(spirv_module.data(), spirv_module.size(), &optimized)) << diagnostics;
        ASSERT_TRUE(tools.Validate(optimized)) << diagnostics;
        ASSERT_TRUE(tools.Disassemble(optimized, &disassembly));
        // Verify that optimization inlined the workgroup-index helper, leaving no function calls.
        EXPECT_THAT(disassembly, ::testing::Not(::testing::HasSubstr("OpFunctionCall")));
        EXPECT_THAT(disassembly, ::testing::HasSubstr("BuiltIn WorkgroupId"));
      }
    }
  }
}

TEST(WorkgroupShader, UnusedHelperIsNotLinked) {
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10000);
  IRBuilder ir(Arch::vulkan, &caps);
  ir.init_header();
  auto main = ir.new_function();
  ir.start_function(main);
  ir.make_inst(spv::OpReturn);
  ir.make_inst(spv::OpFunctionEnd);
  ir.commit_kernel_function(/* func= */ main, /* name= */ "main", /* args= */ {}, /* local_size= */ {1, 1, 1});
  auto spirv_module = ir.finalize();
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_0);
  ASSERT_TRUE(tools.Validate(spirv_module));
  std::string disassembly;
  ASSERT_TRUE(tools.Disassemble(spirv_module, &disassembly));
  EXPECT_THAT(disassembly, ::testing::Not(::testing::HasSubstr("WorkgroupId")));
  EXPECT_THAT(disassembly, ::testing::Not(::testing::HasSubstr("Linkage")));
}

}  // namespace quadrants::lang::spirv
