#include "quadrants/common/logging.h"
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
        std::vector<uint32_t> binary = ir.finalize();
        EXPECT_EQ(binary[1], version);
        spv_target_env environment = version >= 0x10600   ? SPV_ENV_VULKAN_1_3
                                     : version >= 0x10300 ? SPV_ENV_VULKAN_1_1
                                                          : SPV_ENV_VULKAN_1_0;
        spvtools::SpirvTools tools(environment);
        std::string diagnostics;
        spvtools::MessageConsumer report = [&](spv_message_level_t, const char *, const spv_position_t &,
                                               const char *message) {
          diagnostics += message;
          diagnostics += '\n';
        };
        tools.SetMessageConsumer(report);
        ASSERT_TRUE(tools.Validate(binary)) << diagnostics;
        std::string disassembly;
        ASSERT_TRUE(tools.Disassemble(binary, &disassembly));
        EXPECT_NE(disassembly.find("BuiltIn WorkgroupId"), std::string::npos);
        EXPECT_EQ(disassembly.find("LinkageAttributes"), std::string::npos);
        spvtools::Optimizer optimizer(environment);
        optimizer.SetMessageConsumer(report);
        optimizer.RegisterPerformancePasses();
        std::vector<uint32_t> optimized;
        ASSERT_TRUE(optimizer.Run(binary.data(), binary.size(), &optimized)) << diagnostics;
        ASSERT_TRUE(tools.Validate(optimized)) << diagnostics;
        ASSERT_TRUE(tools.Disassemble(optimized, &disassembly));
        EXPECT_EQ(disassembly.find("OpFunctionCall"), std::string::npos);
        EXPECT_NE(disassembly.find("BuiltIn WorkgroupId"), std::string::npos);
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
  ir.commit_kernel_function(main, "main", {}, {1, 1, 1});
  auto binary = ir.finalize();
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_0);
  ASSERT_TRUE(tools.Validate(binary));
  std::string disassembly;
  ASSERT_TRUE(tools.Disassemble(binary, &disassembly));
  EXPECT_EQ(disassembly.find("WorkgroupId"), std::string::npos);
  EXPECT_EQ(disassembly.find("Linkage"), std::string::npos);
}

}  // namespace quadrants::lang::spirv
