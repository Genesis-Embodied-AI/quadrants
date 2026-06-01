#pragma once

#include <cstdint>
#include <vector>

#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/rhi/arch.h"
#include "quadrants/rhi/public_device.h"

namespace quadrants::lang::spirv {

// Builds the SPIR-V compute shader that evaluates the per-task adstack `SizeExpr` bytecode on the device,
// mirroring the LLVM runtime's `runtime_eval_adstack_size_expr`. The shader is compiled once per
// `GfxRuntime` and reused across every reverse-mode kernel launch that has adstack allocas; the bytecode
// buffer, the per-task metadata output buffer, and the kernel arg buffer are rebound per-dispatch through
// the shader's storage-buffer descriptors.
//
// Descriptor set 0:
//   binding 0: readonly  uint32_t[]  - bytecode buffer
//                                      (layout in `quadrants/ir/adstack_size_expr_device.h`)
//   binding 1: rw        uint32_t[]  - per-task metadata output buffer
//                                      `[stride_float, stride_int, (offset_i, max_size_i)*]`
//   binding 2: readonly  uint32_t[]  - kernel arg buffer (source of `ExternalTensorRead` data pointers)
//
// Single-workgroup / single-thread dispatch: the interpreter walks the bytecode iteratively using fixed-size
// value / execution stacks (no recursion - SPIR-V doesn't support it without a non-universal extension).
// `ExternalTensorRead` reads an `uint64_t` data pointer out of the arg buffer at the encoder-precomputed
// `arg_buffer_offset`, then dereferences it through the Physical Storage Buffer path
// (`OpConvertUToPtr` + element access chain), the same PSB path the main kernel uses for ndarray loads - so
// the shader has the same `spirv_has_physical_storage_buffer` capability requirement as the kernels it
// supports. `spirv_has_int64` is also required because the interpreter's value representation is `int64` end
// to end (matches the LLVM runtime's semantics).
//
// Returns the finalised SPIR-V binary as a flat word vector, ready to hand to `Device::create_pipeline`.
// Caller caches the resulting pipeline in `GfxRuntime` and dispatches it per-task before the main kernel.
// Returns an empty vector when the backend does not advertise both capabilities; the caller must then error
// out at launch time rather than silently fall back to host-eval - on a device without PSB + Int64 there is
// no correct way to resolve an `ExternalTensorRead` against a GPU-private ndarray.
std::vector<uint32_t> build_adstack_sizer_spirv(Arch arch, const DeviceCapabilityConfig *caps);

// Maximum number of `SerializedSizeExpr` nodes a single adstack's tree may contribute to the device bytecode.
// The sizer shader's `values_arr` slice of the i64 scratch SSBO is sized to this, indexed by the per-stack
// local offset. Exceeding it on the shader would silently truncate; the host-side encoder must hard-error
// before emitting bytecode past this cap so the failure is attributable to a specific alloca at compile time.
// The cap maps directly into device-memory bytes via `kAdStackSizerScratchI64Elems` (allocated once per
// `GfxRuntime` as a `StorageBuffer` SSBO bound to the sizer dispatch on slot 3): at 65536 i64 entries the
// per-runtime allocation is 512 KiB plus the i32 buffer, negligible against any modern GPU's VRAM budget.
// The previous 4096-cap was chosen to fit the cumulative per-thread private memory budget Blackwell-class
// NVIDIA Vulkan drivers enforce on Function-storage `OpVariable`s; with the scratch state in an SSBO that
// constraint is gone and the cap exists only to keep the encoder's hard-error attached to a fixed compile-
// time ceiling so an unexpectedly deep symbolic tree surfaces as an attributable diagnostic at encode time
// rather than as an out-of-bounds shader access at dispatch time.
constexpr int kAdStackSizerMaxNodesPerStack = 65536;

// Maximum `MaxOverRange` nesting depth the sizer shader can hold on its per-invocation pending-frame stack
// (`pending_*_arr`). The host-side encoder hard-errors when a tree's MOR nesting exceeds this so the shader's
// fixed-size access-chain stays in bounds without a runtime guard. 16 covers every observed real kernel.
constexpr int kAdStackSizerMaxPendingFrames = 16;

// Per-invocation interpreter state element counts in each scratch SSBO. Layout in the i64 buffer is
// `[values_arr | scope_arr | pending_cur_i | pending_end | pending_max_accum]`; layout in the i32 buffer is
// `[pending_mor_idx | pending_body_start | pending_body_end | pending_var_id | pending_saved_max_k]`. The
// host launcher uses these to size the two scratch buffers it binds on every sizer dispatch (binding 3 = i64
// scratch, binding 4 = i32 scratch). Mirrors the in-shader `kI64Base*` / `kI32Base*` constants in
// `adstack_sizer_shader.cpp`.
constexpr int kAdStackSizerScratchI64Elems =
    kAdStackSizerMaxNodesPerStack + kAdStackSizeExprDeviceMaxBoundVars + 3 * kAdStackSizerMaxPendingFrames;
constexpr int kAdStackSizerScratchI32Elems = 5 * kAdStackSizerMaxPendingFrames;

}  // namespace quadrants::lang::spirv
