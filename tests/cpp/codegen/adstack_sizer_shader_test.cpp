// `quadrants/common/logging.h` must come first: it pulls in `<spdlog/fmt/fmt.h>` which declares `fmt::formatter`,
// and `rhi/public_device.h` specialises `fmt::formatter<RhiResult>` without its own include of fmt. Swapping
// the include order here produces a cryptic "use of undeclared identifier 'fmt'" in `public_device.h`.
#include "quadrants/common/logging.h"

#include <cstdio>
#include <fstream>

#include "gtest/gtest.h"
#include "quadrants/codegen/spirv/adstack_sizer_shader.h"
#include "quadrants/rhi/public_device.h"

// Builds the adstack sizer SPIR-V binary with a synthetic capability set that matches a PSB+Int64-capable
// device and writes the word stream to a temporary file. The CI doesn't run `spirv-val` automatically - but
// dumping the binary makes it trivial to validate / disassemble the output during local debugging:
//   spirv-val /tmp/adstack_sizer.spv
//   spirv-dis /tmp/adstack_sizer.spv | head -200
namespace quadrants::lang::spirv {

TEST(AdStackSizerShader, DumpBinary) {
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10400);
  caps.set(DeviceCapability::spirv_has_int8, 1);
  caps.set(DeviceCapability::spirv_has_int16, 1);
  caps.set(DeviceCapability::spirv_has_int64, 1);
  caps.set(DeviceCapability::spirv_has_physical_storage_buffer, 1);

  auto binary = build_adstack_sizer_spirv(Arch::vulkan, &caps);
  ASSERT_FALSE(binary.empty());

  const char *out_path = "/tmp/adstack_sizer.spv";
  std::ofstream f(out_path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(binary.data()), binary.size() * sizeof(uint32_t));
  f.close();
  std::fprintf(stderr, "[adstack_sizer_test] wrote %zu words (%zu bytes) to %s\n", binary.size(),
               binary.size() * sizeof(uint32_t), out_path);
}

}  // namespace quadrants::lang::spirv
