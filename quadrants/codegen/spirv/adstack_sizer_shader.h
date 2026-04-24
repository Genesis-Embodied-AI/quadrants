#pragma once

#include <cstdint>
#include <vector>

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
// The sizer shader's `values_arr` private i64 scratch is sized to this, indexed by the per-stack local offset.
// Exceeding it on the shader would silently truncate; the host-side encoder must hard-error before emitting
// bytecode past this cap so the failure is attributable to a specific alloca at compile time.
constexpr int kAdStackSizerMaxNodesPerStack = 4096;

}  // namespace quadrants::lang::spirv
