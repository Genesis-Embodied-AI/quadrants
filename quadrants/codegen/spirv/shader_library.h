#pragma once

#include <cstdint>
#include <vector>

namespace quadrants::lang::spirv {

std::vector<uint32_t> link_shader_helpers(const std::vector<uint32_t> &kernel);

}  // namespace quadrants::lang::spirv
