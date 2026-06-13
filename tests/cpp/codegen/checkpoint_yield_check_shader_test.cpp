// `quadrants/common/logging.h` must come first: it pulls in `<spdlog/fmt/fmt.h>` which declares `fmt::formatter`, and
// `rhi/public_device.h` specialises `fmt::formatter<RhiResult>` without its own include of fmt. Swapping the include
// order here produces a cryptic "use of undeclared identifier 'fmt'" in `public_device.h`.
#include "quadrants/common/logging.h"

#include <cstdio>
#include <fstream>

#include "gtest/gtest.h"
#include "quadrants/codegen/spirv/checkpoint_yield_check_shader.h"
#include "quadrants/rhi/public_device.h"

// Builds the checkpoint yield-check SPIR-V binary with a vanilla compute-capability set and writes the word stream to
// a temporary file. The CI doesn't run `spirv-val` automatically -- but dumping the binary makes it trivial to
// validate / disassemble the output during local debugging:
//   spirv-val /tmp/checkpoint_yield_check.spv spirv-dis /tmp/checkpoint_yield_check.spv | head -200
// Parity with `adstack_bound_reducer_shader_test.cpp` / `adstack_sizer_shader_test.cpp` -- every SPIR-V shader builder
// in `quadrants/codegen/spirv/` has a matching `DumpBinary` test that guards the IR-builder from regressions which
// would otherwise only surface at runtime on a Vulkan / Metal device.
namespace quadrants::lang::spirv {

TEST(CheckpointYieldCheckShader, DumpBinary) {
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10400);

  auto binary = build_checkpoint_yield_check_spirv(Arch::vulkan, &caps);
  ASSERT_FALSE(binary.empty());

  const char *out_path = "/tmp/checkpoint_yield_check.spv";
  std::ofstream f(out_path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(binary.data()), binary.size() * sizeof(uint32_t));
  f.close();
  std::fprintf(stderr, "[checkpoint_yield_check_test] wrote %zu words (%zu bytes) to %s\n", binary.size(),
               binary.size() * sizeof(uint32_t), out_path);
}

// The yield-check shader uses `OpAtomicCompareExchange` on a u32 in an SSBO, which is in the universally supported
// Vulkan / Metal compute capability set (no extra Atomic-Float / Atomic-Int64 caps). Pins that the builder returns a
// non-empty binary on a minimum-cap caller, matching the header doc-comment guarantee.
TEST(CheckpointYieldCheckShader, NoExtraCapabilityRequired) {
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10400);
  EXPECT_FALSE(build_checkpoint_yield_check_spirv(Arch::vulkan, &caps).empty());
}

}  // namespace quadrants::lang::spirv
