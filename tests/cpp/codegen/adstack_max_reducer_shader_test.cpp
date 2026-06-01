// `quadrants/common/logging.h` must come first: it pulls in `<spdlog/fmt/fmt.h>` which declares `fmt::formatter`, and
// `rhi/public_device.h` specialises `fmt::formatter<RhiResult>` without its own include of fmt. Swapping the include
// order here produces a cryptic "use of undeclared identifier 'fmt'" in `public_device.h`.
#include "quadrants/common/logging.h"

#include <cstdio>
#include <fstream>

#include "gtest/gtest.h"
#include "quadrants/codegen/spirv/adstack_max_reducer_shader.h"
#include "quadrants/rhi/public_device.h"

// Builds the adstack max-reducer SPIR-V binary with a synthetic capability set that matches a PSB+Int64-capable device
// and writes the word stream to a temporary file. The CI does not run `spirv-val` automatically, but dumping the binary
// makes it trivial to validate / disassemble the output during local debugging: spirv-val
// /tmp/adstack_max_reducer.spv spirv-dis /tmp/adstack_max_reducer.spv | head -200
namespace quadrants::lang::spirv {

TEST(AdStackMaxReducerShader, DumpBinary) {
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10400);
  caps.set(DeviceCapability::spirv_has_int64, 1);
  caps.set(DeviceCapability::spirv_has_physical_storage_buffer, 1);

  auto binary = build_adstack_max_reducer_spirv(Arch::vulkan, &caps);
  ASSERT_FALSE(binary.empty());

  const char *out_path = "/tmp/adstack_max_reducer.spv";
  std::ofstream f(out_path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(binary.data()), binary.size() * sizeof(uint32_t));
  f.close();
  std::fprintf(stderr, "[adstack_max_reducer_test] wrote %zu words (%zu bytes) to %s\n", binary.size(),
               binary.size() * sizeof(uint32_t), out_path);
}

// Pins that the two required capabilities are gated at the top of `build_adstack_max_reducer_spirv`: dropping either
// PSB or Int64 flips the return to empty so the dispatch site (`GfxRuntime::dispatch_max_reducers` in
// `runtime/gfx/adstack_max_reducer_launch.cpp`) early-returns an empty result map and the captured `MaxOverRange` falls
// back through the per-task sizer's existing capped path instead of feeding invalid SPIR-V to a pipeline factory that
// would assert at create time. PSB is required because every body leaf reads through the ndarray data pointer the
// kernel arg buffer carries (PSB load); Int64 is required because the body interpreter widens every integer leaf to
// i64 and the begin / per-axis-begin reassembly arithmetic uses 64-bit operations. The output atomic itself is u32 so
// no atomic-i64 capability is needed.
TEST(AdStackMaxReducerShader, GateReturnsEmptyWhenRequiredCapIsMissing) {
  auto make_caps = []() {
    DeviceCapabilityConfig caps;
    caps.set(DeviceCapability::spirv_version, 0x10400);
    caps.set(DeviceCapability::spirv_has_int64, 1);
    caps.set(DeviceCapability::spirv_has_physical_storage_buffer, 1);
    return caps;
  };

  {
    auto caps = make_caps();
    caps.set(DeviceCapability::spirv_has_physical_storage_buffer, 0);
    EXPECT_TRUE(build_adstack_max_reducer_spirv(Arch::vulkan, &caps).empty());
  }
  {
    auto caps = make_caps();
    caps.set(DeviceCapability::spirv_has_int64, 0);
    EXPECT_TRUE(build_adstack_max_reducer_spirv(Arch::vulkan, &caps).empty());
  }
  // Sanity: all required caps present still builds a non-empty binary.
  {
    auto caps = make_caps();
    EXPECT_FALSE(build_adstack_max_reducer_spirv(Arch::vulkan, &caps).empty());
  }
}

}  // namespace quadrants::lang::spirv
