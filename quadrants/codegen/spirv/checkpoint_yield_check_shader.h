#pragma once

#include <cstdint>
#include <vector>

#include <spdlog/fmt/fmt.h>

#include "quadrants/rhi/arch.h"
#include "quadrants/rhi/public_device.h"

namespace quadrants::lang::spirv {

// Builds the SPIR-V compute shader that implements GPU-side yield detection for `qd.checkpoint(yield_on=...)` blocks on
// Vulkan / Metal. One yield-check dispatch runs in the same cmdlist as the checkpoint's body kernels, immediately after
// the last body kernel. It reads the user's `yield_on=` flag and, if non-zero, atomic-CASes the checkpoint id into the
// shared `yield_signal` slot (first-yielder-wins, mirroring the CUDA SM 9.0+ semantics in
// `runtime/cuda/checkpoint_yield_check.cu`). The framework deliberately does NOT clear the user's flag here -- the
// host loop owns that buffer and is responsible for clearing it before the resume launch.
//
// The yield-check dispatch is itself indirect-dispatched off the same per-kernel dim3 slot the gate shader wrote, so
// a skipped checkpoint also skips its yield-check (no atomic ops on a checkpoint that did not run).
//
// Bindings (descriptor set 0):
//   0: rw       uint32_t[2]      - control:  [resume_point: i32, yield_signal: i32]
//                                            (shared with `checkpoint_gate_shader`; see `CheckpointControlBuf` for the
//                                            layout)
//   1: rw       uint32_t[1]      - yield_on: the user's `yield_on=` ndarray, single i32 element
//                                            (treated as readonly by the shader; we keep the binding rw to avoid
//                                            churning the descriptor layout)
//   2: readonly uint32_t[1]      - params:   [cp_id: u32]
//
// Dispatched once per yielding checkpoint at (1, 1, 1).
//
// Returns the finalized SPIR-V binary; never empty (vanilla compute pipeline, no extra capability requirements beyond
// OpAtomicCompareExchange which is universally supported on the Quadrants-targeted Vulkan / Metal surface).
std::vector<uint32_t> build_checkpoint_yield_check_spirv(Arch arch, const DeviceCapabilityConfig *caps);

// Word offsets within the params SSBO (binding 2). Layout fixed; host populates once at graph-build time matching these
// offsets.
struct CheckpointYieldCheckParams {
  static constexpr uint32_t kWordOffsetCpId = 0;
  static constexpr uint32_t kNumWords = 1;
};

}  // namespace quadrants::lang::spirv
