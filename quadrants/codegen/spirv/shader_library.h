#pragma once

#include <cstdint>
#include <vector>

namespace quadrants::lang::spirv {

std::vector<uint32_t> link_workgroup_helper(const std::vector<uint32_t> &kernel);

}  // namespace quadrants::lang::spirv
