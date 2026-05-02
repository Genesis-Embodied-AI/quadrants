// `quadrants/common/logging.h` must come first: it pulls in `<spdlog/fmt/fmt.h>` which declares `fmt::formatter`, and
// `rhi/public_device.h` specialises `fmt::formatter<RhiResult>` without its own include of fmt. Swapping the include
// order here produces a cryptic "use of undeclared identifier 'fmt'" in `public_device.h`.
#include "quadrants/common/logging.h"

#include <cstdio>
#include <fstream>

#include "gtest/gtest.h"
#include "quadrants/codegen/spirv/adstack_bound_reducer_shader.h"
#include "quadrants/rhi/public_device.h"

// Builds the adstack bound-reducer SPIR-V binary with a synthetic capability set that matches a PSB+Int64-capable
// device and writes the word stream to a temporary file. The CI doesn't run `spirv-val` automatically - but dumping the
// binary makes it trivial to validate / disassemble the output during local debugging: spirv-val
// /tmp/adstack_bound_reducer.spv spirv-dis /tmp/adstack_bound_reducer.spv | head -200
namespace quadrants::lang::spirv {

TEST(AdStackBoundReducerShader, DumpBinary) {
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10400);
  caps.set(DeviceCapability::spirv_has_int64, 1);
  caps.set(DeviceCapability::spirv_has_physical_storage_buffer, 1);

  auto binary = build_adstack_bound_reducer_spirv(Arch::vulkan, &caps);
  ASSERT_FALSE(binary.empty());

  const char *out_path = "/tmp/adstack_bound_reducer.spv";
  std::ofstream f(out_path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(binary.data()), binary.size() * sizeof(uint32_t));
  f.close();
  std::fprintf(stderr, "[adstack_bound_reducer_test] wrote %zu words (%zu bytes) to %s\n", binary.size(),
               binary.size() * sizeof(uint32_t), out_path);
}

// Same as DumpBinary but with the `spirv_has_float64` capability also set, so the f64-comparison arm of the shader is
// emitted. Pins that the f64 extension path builds without rejecting at IR-builder level on a cap-OK device. The host
// launcher's filter (`adstack_bound_reducer_launch.cpp`) drops f64-captured `bound_expr`s on devices that do not set
// this cap, so the f64 arm only runs when the cap is present; the test verifies the shader itself is well-formed under
// that cap combination.
TEST(AdStackBoundReducerShader, DumpBinaryWithFloat64) {
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10400);
  caps.set(DeviceCapability::spirv_has_int64, 1);
  caps.set(DeviceCapability::spirv_has_float64, 1);
  caps.set(DeviceCapability::spirv_has_physical_storage_buffer, 1);

  auto binary = build_adstack_bound_reducer_spirv(Arch::vulkan, &caps);
  ASSERT_FALSE(binary.empty());
}

// Pins that the two required capabilities are gated at the top of `build_adstack_bound_reducer_spirv`: dropping either
// PSB or Int64 flips the return to empty so the launcher's matching `flush()`+`wait_idle()` early-return at
// `adstack_bound_reducer_launch.cpp` surfaces a "legacy device missing a required hardware feature" outcome (heap stays
// at the dispatched-threads worst case) instead of emitting invalid SPIR-V. PSB-less or Int64-less devices cannot run
// the shader because the host-side parameter blob the shader consumes via `OpLoad` of a `restrict Aliased` PSB pointer
// requires both caps; Float64 is NOT a required cap because the f64 arm is conditional inside the shader and the f32 /
// i32 arms work on every device.
TEST(AdStackBoundReducerShader, GateReturnsEmptyWhenRequiredCapIsMissing) {
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
    EXPECT_TRUE(build_adstack_bound_reducer_spirv(Arch::vulkan, &caps).empty());
  }
  {
    auto caps = make_caps();
    caps.set(DeviceCapability::spirv_has_int64, 0);
    EXPECT_TRUE(build_adstack_bound_reducer_spirv(Arch::vulkan, &caps).empty());
  }
  // Sanity: all caps present still builds a non-empty binary.
  {
    auto caps = make_caps();
    EXPECT_FALSE(build_adstack_bound_reducer_spirv(Arch::vulkan, &caps).empty());
  }
  // Sanity: Float64 is NOT required - dropping it must still produce a valid binary (the shader's f64 arm is dead-code
  // on the device, but the f32 / i32 arms remain functional).
  {
    auto caps = make_caps();
    caps.set(DeviceCapability::spirv_has_float64, 0);
    EXPECT_FALSE(build_adstack_bound_reducer_spirv(Arch::vulkan, &caps).empty());
  }
}

}  // namespace quadrants::lang::spirv
