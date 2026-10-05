#include "quadrants/common/logging.h"
#include "gtest/gtest.h"
#include "quadrants/codegen/spirv/spirv_ir_builder.h"
#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/optimizer.hpp"

namespace quadrants::lang::spirv {

TEST(WorkgroupShader, LinksAndValidatesAcrossTargets) {
  for (auto arch : {Arch::vulkan, Arch::metal}) {
    for (auto version : {0x10000, 0x10300, 0x10600}) {
      for (bool physical : {false, true}) {
        if (physical && (arch == Arch::metal || version < 0x10300)) {
          continue;
        }
        SCOPED_TRACE(fmt::format("arch={} version={} physical={}", arch_name(arch), version, physical));
        DeviceCapabilityConfig caps;
        caps.set(DeviceCapability::spirv_version, version);
        caps.set(DeviceCapability::spirv_has_physical_storage_buffer, physical);
        IRBuilder ir(arch, &caps);
        ir.init_header();
        auto output = ir.buffer_argument(ir.u32_type(), 0, 0, "result");
        auto main = ir.new_function();
        ir.start_function(main);
        for (uint32_t dim : {0u, 1u, 2u, 0u}) {
          auto index = ir.get_work_group_id(dim);
          auto destination =
              ir.struct_array_access(ir.u32_type(), output, ir.uint_immediate_number(ir.u32_type(), dim));
          auto comparison = ir.ge(ir.cast(ir.i32_type(), index), ir.int_immediate_number(ir.i32_type(), 1));
          ir.store_variable(destination, ir.cast(ir.u32_type(), comparison));
        }
        ir.make_inst(spv::OpReturn);
        ir.make_inst(spv::OpFunctionEnd);
        ir.commit_kernel_function(main, "main", {output}, {1, 1, 1});
        auto binary = ir.finalize();
        EXPECT_EQ(binary[1], version);
        auto environment = version >= 0x10600   ? SPV_ENV_VULKAN_1_3
                           : version >= 0x10300 ? SPV_ENV_VULKAN_1_1
                                                : SPV_ENV_VULKAN_1_0;
        spvtools::SpirvTools tools(environment);
        std::string diagnostics;
        auto report = [&](spv_message_level_t, const char *, const spv_position_t &, const char *message) {
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

TEST(WorkgroupShader, ComparisonWithoutWorkgroupQuery) {
  for (auto arch : {Arch::vulkan, Arch::metal}) {
    DeviceCapabilityConfig caps;
    caps.set(DeviceCapability::spirv_version, 0x10000);
    IRBuilder ir(arch, &caps);
    ir.init_header();
    auto buffer = ir.buffer_argument(ir.i32_type(), 0, 0, "values");
    auto main = ir.new_function();
    ir.start_function(main);
    auto address = ir.struct_array_access(ir.i32_type(), buffer, ir.int_immediate_number(ir.i32_type(), 0));
    auto value = ir.load_variable(address, ir.i32_type());
    auto result = ir.ge(value, ir.int_immediate_number(ir.i32_type(), -1));
    ir.store_variable(address, ir.cast(ir.i32_type(), result));
    ir.make_inst(spv::OpReturn);
    ir.make_inst(spv::OpFunctionEnd);
    ir.commit_kernel_function(main, "main", {buffer}, {1, 1, 1});
    auto binary = ir.finalize();
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_0);
    ASSERT_TRUE(tools.Validate(binary));
    std::string disassembly;
    ASSERT_TRUE(tools.Disassemble(binary, &disassembly));
    EXPECT_NE(disassembly.find("OpFunctionCall"), std::string::npos);
    EXPECT_EQ(disassembly.find("LinkageAttributes"), std::string::npos);
    spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_0);
    optimizer.RegisterPerformancePasses();
    std::vector<uint32_t> optimized;
    ASSERT_TRUE(optimizer.Run(binary.data(), binary.size(), &optimized));
    ASSERT_TRUE(tools.Validate(optimized));
    ASSERT_TRUE(tools.Disassemble(optimized, &disassembly));
    EXPECT_EQ(disassembly.find("OpFunctionCall"), std::string::npos);
    EXPECT_NE(disassembly.find("OpSGreaterThanEqual"), std::string::npos);
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
