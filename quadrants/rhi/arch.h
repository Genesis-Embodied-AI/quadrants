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

// Wavefront / warp / subgroup size for the GPU backends quadrants compiles for. CUDA is fixed at 32 hardware-wide;
// AMDGPU is forced to 64 on every target (see hp/always-wave64 — RDNA defaults to wave32 in LLVM's AMDGPU backend, but
// we override target-features and link the wavefrontsize64=on libdevice variant to standardize on wave64). Returns 0
// for non-LLVM-GPU backends (Vulkan / Metal query subgroup size at runtime through SPIR-V) and for CPU. Also folded
// into the offline-cache key so cached kernels invalidate if this constant ever changes.
constexpr int kCudaWarpSize = 32;
constexpr int kAmdgpuWaveSize = 64;

int subgroup_size(Arch arch);

}  // namespace quadrants
