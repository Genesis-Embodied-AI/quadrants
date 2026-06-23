#pragma once

#include <cstdint>
#include <vector>

#include <spdlog/fmt/fmt.h>

#include "quadrants/rhi/arch.h"
#include "quadrants/rhi/public_device.h"

namespace quadrants::lang::spirv {

// Builds the SPIR-V compute shader that implements GPU-side per-checkpoint gating for `qd.checkpoint(...)` blocks on
// Vulkan / Metal. One gate dispatch runs in the same cmdlist before each checkpoint's body kernels. It reads
// `(resume_point, yield_signal)` from the per-launch control SSBO and writes either the active workgroup-count triple
// or `(0, 0, 0)` into each body kernel's slot of the shared dim3 buffer; body kernels then issue via
// `CommandList::dispatch_indirect` so a skipped checkpoint dispatches zero workgroups (no GPU work beyond the
// indirect-dispatch issue itself, ~1 us / kernel).
//
// Mirror of the CUDA SM 9.0+ `_qd_checkpoint_if_gate` kernel in `runtime/cuda/checkpoint_gate.cu`, adapted for the
// indirect-dispatch backends that lack conditional-graph-node hardware. The CUDA path can flip a single conditional
// handle; we have to write per-body-kernel grid dims because indirect dispatch is the only device-side
// dispatch-mode primitive Vulkan / Metal expose. See `perso_hugh/doc/qipc/reentrant.md` section 6.2 for the design.
//
// Compiled once per `GfxRuntime` (generic in cp_id / n_kernels / active dims) and reused across every yielding-capable
// kernel launch. Per-checkpoint specialization is carried in the params SSBO bound at slot 1, populated at graph-build
// time and never written by the GPU.
//
// Bindings (descriptor set 0):
//   0: rw       uint32_t[2]      - control: [resume_point: i32, yield_signal: i32]
//   1: readonly uint32_t[2+3*N]  - params:  [cp_id: u32, n_kernels: u32,
//                                            (gx_i, gy_i, gz_i) per body kernel]
//   2: rw       uint32_t[3*N]    - out_dims: (gx_i, gy_i, gz_i) per body kernel; written each
//                                            launch by the gate, consumed by each body kernel's
//                                            `CommandList::dispatch_indirect` from offset 12*i.
//
// Dispatched once per checkpoint at (1, 1, 1). N is small (handful to ~30 body kernels per checkpoint), so the
// single-thread loop fits well inside one workgroup.
//
// Returns the finalized SPIR-V binary; never empty (no capability requirement beyond a vanilla compute pipeline).
// Caller compiles into a `Pipeline` via `Device::create_pipeline`.
std::vector<uint32_t> build_checkpoint_gate_spirv(Arch arch, const DeviceCapabilityConfig *caps);

// Word offsets within the params SSBO (binding 1) the gate shader reads. Layout is fixed; the host launcher writes the
// params buffer once at graph-build time matching these offsets.
struct CheckpointGateParams {
  static constexpr uint32_t kWordOffsetCpId = 0;
  static constexpr uint32_t kWordOffsetNKernels = 1;
  static constexpr uint32_t kWordOffsetDimsBase = 2;  // (gx, gy, gz) tuples follow, 3 u32 words per kernel
};

// Word offsets within the control SSBO (binding 0). Layout is shared with the yield-check shader in
// `checkpoint_yield_check_shader.h`; the two shaders operate on the same per-launch control buffer.
struct CheckpointControlBuf {
  static constexpr uint32_t kWordOffsetResumePoint = 0;
  static constexpr uint32_t kWordOffsetYieldSignal = 1;
  static constexpr uint32_t kNumWords = 2;
};

}  // namespace quadrants::lang::spirv
