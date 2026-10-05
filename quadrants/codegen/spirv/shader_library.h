#pragma once

#include <cstdint>
#include <vector>

namespace quadrants::lang::spirv {

std::vector<uint32_t> link_shader_helpers(const std::vector<uint32_t> &kernel, bool use_int64_helpers);

}  // namespace quadrants::lang::spirv
