#include "quadrants/common/logging.h"

#include "gtest/gtest.h"
#include "spirv-tools/libspirv.hpp"
#ifdef QD_WITH_METAL
#include "spirv_msl.hpp"
#endif
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

// Verify that the buffer code retained after dropping SPIR-V <1.5 still generates valid shaders.
TEST(SpirvVersion, ValidatesBuffersAndGlobalInterfaces) {
  // Vulkan GLSL illustration of the buffer accesses; the test also exercises built-ins, RNG, and workgroup memory.
  // #version 450
  // layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
  // layout(std430, set = 0, binding = 0) buffer ArrayBuffer { uint elements[]; } array_buffer;
  // layout(std430, set = 0, binding = 1) buffer ScalarBuffer { uint value; } scalar_buffer;
  // layout(std140, set = 0, binding = 2) uniform ScalarUniform { uint value; } scalar_uniform;
  //
  // void main() {
  //     uint v = scalar_uniform.value;  // Exercise reading a uniform-buffer member.
  //     array_buffer.elements[0] = v;   // Exercise writing a storage-buffer array element.
  //     scalar_buffer.value = v;       // Exercise writing a storage-buffer struct member.
  // }
  for (Arch arch : {Arch::vulkan, Arch::metal}) {
    for (uint32_t version : {0x10500u, 0x10600u}) {
      SCOPED_TRACE(version);
      DeviceCapabilityConfig caps;
      caps.set(DeviceCapability::spirv_version, version);
      caps.set(DeviceCapability::spirv_has_subgroup_basic, 1);
      IRBuilder ir(arch, &caps);
      SpirvOperations ops(ir);
      ir.init_header();

      // layout(std430, set = 0, binding = 0) buffer ArrayBuffer { uint elements[]; } array_buffer;
      Value array_buffer = ir.buffer_argument(/* value_type= */ ir.u32_type(), /* descriptor_set= */ 0,
                                              /* binding= */ 0, /* name= */ "array_buffer");
      // Each tuple contains the member's SPIR-V type, name, and byte offset within the struct.
      std::vector<std::tuple<SType, std::string, size_t>> struct_members = {{ir.u32_type(), "value", 0}};
      // layout(std430, set = 0, binding = 1) buffer ScalarBuffer { uint value; } scalar_buffer;
      Value scalar_buffer =
          ir.buffer_struct_argument(/* struct_type= */ ir.create_struct_type(struct_members),
                                    /* descriptor_set= */ 0, /* binding= */ 1, /* name= */ "scalar_buffer");
      // layout(std140, set = 0, binding = 2) uniform ScalarUniform { uint value; } scalar_uniform;
      Value scalar_uniform =
          ir.uniform_struct_argument(/* struct_type= */ ir.create_struct_type(struct_members),
                                     /* descriptor_set= */ 0, /* binding= */ 2, /* name= */ "scalar_uniform");
      Value shared = ir.alloca_workgroup_array(ir.get_function_array_type(ir.u32_type(), 1));
      SType storage_ptr_type = ir.get_storage_pointer_type(ir.u32_type());
      SType uniform_ptr_type = ir.get_pointer_type(ir.u32_type(), spv::StorageClassUniform);
      EXPECT_EQ(storage_ptr_type.storage_class, spv::StorageClassStorageBuffer);
      EXPECT_EQ(scalar_uniform.stype.storage_class, spv::StorageClassUniform);

      Value main = ir.new_function();
      ir.start_function(main);
      Value zero = ir.uint_immediate_number(ir.u32_type(), 0);

      // uint v = scalar_uniform.value;
      Value uniform_ptr = ir.make_value(/* op= */ spv::OpAccessChain, /* out_type= */ uniform_ptr_type,
                                        /* base= */ scalar_uniform, /* member_index= */ zero);
      Value v = ir.load_variable(uniform_ptr, ir.u32_type());
      // Exercise both input-registration paths and the private globals used by the random-number generator.
      v = ops.add(v, ops.get_global_invocation_id(0));
      v = ops.add(v, ops.get_local_invocation_id(0));
      v = ops.add(v, ops.get_subgroup_invocation_id());
      v = ops.add(v, ops.rand_u32(array_buffer));
      SType shared_ptr_type = ir.get_pointer_type(ir.u32_type(), spv::StorageClassWorkgroup);
      Value shared_ptr = ir.make_value(spv::OpAccessChain, shared_ptr_type, shared, zero);
      ir.store_variable(shared_ptr, v);
      v = ir.load_variable(shared_ptr, ir.u32_type());

      // array_buffer.elements[0] = v;
      ir.store_variable(
          ir.struct_array_access(/* res_type= */ ir.u32_type(), /* buffer= */ array_buffer, /* index= */ zero), v);

      // scalar_buffer.value = v;
      Value scalar_ptr =
          ir.make_value(/* op= */ spv::OpAccessChain, /* out_type= */ storage_ptr_type, scalar_buffer, zero);
      ir.store_variable(scalar_ptr, v);
      ir.make_inst(spv::OpReturn);
      ir.make_inst(spv::OpFunctionEnd);

      // SPIR-V 1.5+ entry-point interfaces include the buffers and workgroup variable used by this function.
      std::vector<Value> entry_point_args = {array_buffer, scalar_buffer, scalar_uniform, shared};
      ir.commit_kernel_function(/* func= */ main, /* name= */ "main", /* args= */ entry_point_args,
                                /* local_size= */ {1, 1, 1});
      std::vector<uint32_t> spirv_module = ir.finalize();
      ASSERT_GT(spirv_module.size(), 5);
      EXPECT_EQ(spirv_module[1], version);

      spv_target_env env;
      // clang-format off
      switch (version) {
        case 0x10500: env = SPV_ENV_VULKAN_1_2; break;
        case 0x10600: env = SPV_ENV_VULKAN_1_3; break;
        default: GTEST_FAIL() << "Unexpected SPIR-V version: " << version;
      }
      // clang-format on
      spvtools::SpirvTools tools(env);
      std::string diagnostics;
      tools.SetMessageConsumer([&](spv_message_level_t, const char *, const spv_position_t &, const char *message) {
        diagnostics += message;
        diagnostics += '\n';
      });
      EXPECT_TRUE(tools.Validate(spirv_module)) << diagnostics;

#ifdef QD_WITH_METAL
      if (arch == Arch::metal) {
        // SPIRV-Cross hides globals omitted from a 1.4+ interface. Check Metal translation in Metal-enabled builds.
        spirv_cross::CompilerMSL compiler(spirv_module);
        spirv_cross::CompilerMSL::Options options;
        options.enable_decoration_binding = true;
        options.set_msl_version(2, 1, 0);
        compiler.set_msl_options(options);
        std::string msl = compiler.compile();
        EXPECT_FALSE(msl.empty());
        spirv_cross::ShaderResources resources = compiler.get_shader_resources();
        EXPECT_EQ(resources.storage_buffers.size(), 2);
        EXPECT_EQ(resources.uniform_buffers.size(), 1);
      }
#endif
    }
  }
}

}  // namespace quadrants::lang::spirv
