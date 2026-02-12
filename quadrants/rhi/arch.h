#pragma once

#include <string>

namespace quadrants {

enum class Arch : int {
#define PER_ARCH(x) x,
#include "quadrants/inc/archs.inc.h"

#undef PER_ARCH
};

std::string arch_name(Arch arch);

Arch arch_from_name(const std::string &arch);

bool arch_is_cpu(Arch arch);

bool arch_is_cuda(Arch arch);

bool arch_is_amdgpu(Arch arch);

bool arch_is_metal(Arch arch);

bool arch_uses_llvm(Arch arch);

bool arch_is_gpu(Arch arch);

bool arch_uses_spirv(Arch arch);

Arch host_arch();

bool arch_use_host_memory(Arch arch);

int default_simd_width(Arch arch);

}  // namespace quadrants
