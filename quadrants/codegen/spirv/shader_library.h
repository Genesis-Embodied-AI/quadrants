#pragma once

#include <cstdint>
#include <vector>

namespace quadrants::lang::spirv {

// Return the compiled workgroup library with an addressing model matching the kernel.
std::vector<uint32_t> get_workgroup_shader_library(const std::vector<uint32_t> &kernel);

// Link the supplied SPIR-V modules and return the combined module.
std::vector<uint32_t> link_spirv_modules(const std::vector<std::vector<uint32_t>> &modules);

}  // namespace quadrants::lang::spirv
