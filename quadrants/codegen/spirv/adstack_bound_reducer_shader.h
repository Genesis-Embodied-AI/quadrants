#pragma once

#include <cstdint>
#include <vector>

#include <spdlog/fmt/fmt.h>

#include "quadrants/rhi/arch.h"
#include "quadrants/rhi/public_device.h"

namespace quadrants::lang::spirv {

// Builds the SPIR-V compute shader that evaluates a captured `TaskAttributes::StaticBoundExpr` predicate over a thread
// range and atomic-adds 1 into a per-task slot of `BufferType::AdStackRowCounter` for each thread that passes.
// Dispatched once per adstack-bearing task before the main task on the static-IR-bound sparse-adstack-heap path; the
// resulting count sizes the float adstack heap allocation exactly.
//
// The shader is generic (parametrised at dispatch time by the parameter blob in binding 2) and is compiled once per
// `GfxRuntime`. Host responsibility per dispatch:
//   - Write the parameter blob (`AdStackBoundReducerParams` below) into a small storage buffer and bind to
//     slot 2.
//   - Bind the kernel arg buffer to slot 0 (the same arg buffer the main kernel uses).
//   - Bind the per-kernel `AdStackRowCounter` to slot 1 with the matching `task_id_in_kernel` slot cleared.
//   - Dispatch `ceil(length / kWorkgroupSize)` work groups of `kWorkgroupSize` threads each.
// After dispatch + sync the slot's value equals the number of threads whose `field[i] cmp threshold` matched the
// captured polarity; the host reads that count and sizes the float heap to `count * stride_float * sizeof(f32)` before
// binding the main task.
//
// Required device capabilities: `spirv_has_physical_storage_buffer` + `spirv_has_int64`. The first is needed because
// the gating field is read through the ndarray data pointer the kernel arg buffer carries (PSB load path, mirroring the
// main kernel's ndarray access); the second is needed for u64 pointer arithmetic. On devices without either capability
// the function returns an empty vector and the runtime falls back to the dispatched-threads worst-case heap sizing
// -safe but no savings.
std::vector<uint32_t> build_adstack_bound_reducer_spirv(Arch arch, const DeviceCapabilityConfig *caps);

// Compute-shader workgroup size (x dimension; y and z are 1). Power-of-two and a multiple of typical subgroup widths on
// Metal / Vulkan so atomic-add contention amortises per workgroup. Host launcher uses this to compute `num_workgroups_x
// = (length + kAdStackBoundReducerWorkgroupSize - 1) / kAdStackBoundReducerWorkgroupSize` per dispatch.
constexpr uint32_t kAdStackBoundReducerWorkgroupSize = 128;

// Layout of the parameter blob the host writes into binding 2 before each dispatch. POD; keep field order in sync with
// the shader's compile-time word-offset constants in `adstack_bound_reducer_shader.cpp`.
struct AdStackBoundReducerParams {
  // Slot index in the per-kernel `BufferType::AdStackRowCounter` array that the matching atomic-adds will accumulate
  // into. Matches the `task_id_in_kernel` of the main task this reducer is sizing.
  uint32_t task_id_in_kernel;
  // Number of threads to dispatch over (the iteration bound of the gating predicate). Threads with
  // `gl_GlobalInvocationID.x >= length` early-return so dispatch can be rounded up to the workgroup-size multiple
  // without overcounting.
  uint32_t length;
  // u32 word offset into the kernel arg buffer where the ndarray data pointer (u64, two adjacent u32 words) lives. The
  // shader does `OpConvertUToPtr` on that pointer and PSB-loads the gating field's element at
  // `gl_GlobalInvocationID.x`. Only used when `field_source_kind == NdArray`; SNode-backed sources are not yet
  // supported by this shader (the runtime's caller falls back to worst-case sizing on SNode).
  uint32_t arg_word_offset;
  // Encodes the captured `StaticBoundExpr::cmp_op` as an integer: 0 = cmp_lt, 1 = cmp_le, 2 = cmp_gt, 3 = cmp_ge, 4 =
  // cmp_eq, 5 = cmp_ne. The shader uses a switch over this code to emit the right SPIR-V comparison op.
  uint32_t op_code;
  // 1 when the gating field's element type is f32 / f64 (the threshold and the loaded element are bitcast to float for
  // the comparison); 0 when the element type is i32 (signed integer comparison). Other types fall back to worst-case
  // sizing in the runtime caller. Combine with `field_dtype_is_double` to pick element width (4 vs 8 bytes) and the f32
  // / f64 comparison arm.
  uint32_t field_dtype_is_float;
  // 1 when the gate enters on the predicate holding (typical `if cmp:` shape); 0 when it sits inside the `else` branch
  // and the predicate must be inverted before counting. The shader applies the polarity flip via XOR after the
  // comparison so the captured count always matches threads that reach the LCA block.
  uint32_t polarity;
  // Low 32 bits of the captured threshold literal. Reinterpreted as f32 when `field_dtype_is_float == 1` and
  // `field_dtype_is_double == 0`, as i32 when `field_dtype_is_float == 0`. f64 thresholds use the
  // `(threshold_bits_high, threshold_bits)` pair (low half here, high half below). Stored in the parameter blob rather
  // than embedded as a SPIR-V `OpConstant` because the shader is compiled once per `GfxRuntime` and the threshold
  // varies per kernel.
  uint32_t threshold_bits;
  // 0 when the gating field comes from a kernel ndarray argument (resolved via the kernel arg buffer + Physical Storage
  // Buffer load); 1 when it comes from an SNode-backed `qd.field(...)` placed under `qd.root.dense(...)` (resolved via
  // a direct word load from the bound root buffer at byte offset `snode_byte_base_offset + gid *
  // snode_byte_cell_stride`). The two paths are mutually exclusive per dispatch.
  uint32_t field_source_is_snode;
  // Byte offset within the bound root buffer of the gating field's first cell value. Equals
  // `dense_snode.mem_offset_in_parent_cell + leaf_snode.mem_offset_in_parent_cell` (precomputed by the IR pattern
  // matcher from the snode descriptor's prefix sums). Read only when `field_source_is_snode == 1`.
  uint32_t snode_byte_base_offset;
  // Stride per `gid` step in bytes for SNode-backed gates - the dense parent's `cell_stride`. The shader walks the
  // gating field via `byte_offset = snode_byte_base_offset + gid * snode_byte_cell_stride` and loads either one u32
  // word (i32 / f32 element type) or two adjacent u32 words (f64 element type). Read only when `field_source_is_snode
  // == 1`.
  uint32_t snode_byte_cell_stride;
  // 1 when the gating field's element type is f64 (the source ndarray / SNode cell stride is 8 bytes per element). The
  // shader walks elements with a doubled byte stride and reassembles the two adjacent u32 words into a u64 -> bitcast
  // f64 for the comparison. Read only when `field_dtype_is_float == 1`; 0 for i32 and f32 gates.
  uint32_t field_dtype_is_double;
  // High 32 bits of an f64 threshold, valid only when `field_dtype_is_double == 1`. The shader reassembles the 64-bit
  // bit pattern from `(threshold_bits_high << 32) | threshold_bits` and bitcasts to f64.
  uint32_t threshold_bits_high;

  // Offset into the parameter blob (in u32 words) for each field; published to the shader and the host launcher as
  // compile-time constants so each side reads/writes the same slots without a separate header serialisation step.
  static constexpr uint32_t kWordOffsetTaskId = 0;
  static constexpr uint32_t kWordOffsetLength = 1;
  static constexpr uint32_t kWordOffsetArgWordOffset = 2;
  static constexpr uint32_t kWordOffsetOpCode = 3;
  static constexpr uint32_t kWordOffsetFieldDtypeIsFloat = 4;
  static constexpr uint32_t kWordOffsetPolarity = 5;
  static constexpr uint32_t kWordOffsetThresholdBits = 6;
  static constexpr uint32_t kWordOffsetFieldSourceIsSnode = 7;
  static constexpr uint32_t kWordOffsetSnodeByteBaseOffset = 8;
  static constexpr uint32_t kWordOffsetSnodeByteCellStride = 9;
  static constexpr uint32_t kWordOffsetFieldDtypeIsDouble = 10;
  static constexpr uint32_t kWordOffsetThresholdBitsHigh = 11;
  static constexpr uint32_t kNumWords = 12;
};

// Op-code values written into `AdStackBoundReducerParams::op_code`. Kept as a free enum (not a class enum) so the host
// launcher can assign directly from `BinaryOpType` without a static_cast.
enum AdStackBoundReducerOpCode : uint32_t {
  kAdStackBoundReducerOpLt = 0,
  kAdStackBoundReducerOpLe = 1,
  kAdStackBoundReducerOpGt = 2,
  kAdStackBoundReducerOpGe = 3,
  kAdStackBoundReducerOpEq = 4,
  kAdStackBoundReducerOpNe = 5,
};

}  // namespace quadrants::lang::spirv
