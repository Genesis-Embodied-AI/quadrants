#pragma once

#include <cstdint>
#include <vector>

#include <spdlog/fmt/fmt.h>

#include "quadrants/rhi/arch.h"
#include "quadrants/rhi/public_device.h"

namespace quadrants::lang::spirv {

// Builds the SPIR-V compute shader that evaluates a captured `StaticAdStackMaxReducerSpec`'s body subtree over a thread
// range and atomic-maxes the result into a per-spec slot of `BufferType::AdStackMaxReducerOutput`. Dispatched once per
// captured `MaxOverRange` node before the main task on the option-D max-reducer path; the resulting per-spec value is
// substituted as a `Const` into the per-stack `SerializedSizeExpr` tree by `substitute_precomputed_max_over_range`
// before any of the three eval paths (host fast path, SPIR-V on-device sizer, LLVM device sizer) walks it.
//
// The shader is generic (parametrised at dispatch time by the parameter blob in binding 2 + the body bytecode in
// binding 3) and is compiled once per `GfxRuntime`. Host responsibility per dispatch:
//   - Write the parameter blob (`AdStackMaxReducerParams` below) into the shared params storage buffer at the spec's
//     descriptor-aligned offset, bound to slot 2 with a per-spec `VkDescriptorBufferInfo::offset`.
//   - Encode the body subtree into the shared bytecode storage buffer at the spec's `body_bytecode_offset_words` slot
//     using the existing `AdStackSizeExprDeviceNode` POD format from `quadrants/ir/adstack_size_expr_device.h`. Bind to
//     slot 3.
//   - Bind the kernel arg buffer to slot 0 (the same arg buffer the main kernel uses) so `kExternalTensorRead` body
//     leaves can resolve their ndarray data pointers via the same byte-offset convention the main kernel uses.
//   - Bind the per-kernel `AdStackMaxReducerOutput` u64 buffer to slot 1 with the matching `output_slot` cleared.
//   - Dispatch `ceil(length / kAdStackMaxReducerWorkgroupSize)` workgroups of `kAdStackMaxReducerWorkgroupSize`
//   threads.
// After dispatch + sync the slot's value equals `max over i in [begin, begin + length): body[i]` interpreted in i64;
// the host reads that value and substitutes it as a `Const` into the per-stack `SizeExpr` tree.
//
// Required device capabilities: `spirv_has_physical_storage_buffer` + `spirv_has_int64`. The first is needed because
// every body leaf reads through the ndarray data pointer the kernel arg buffer carries (PSB load path, mirroring the
// main kernel's ndarray access); the second is needed for u64 atomic-max into the output buffer. On devices missing
// either capability the function returns an empty vector and the runtime hard-errors at dispatch-time
// (`adstack_max_reducer_launch.cpp`'s `QD_ERROR_IF` gate). Silently falling through to the per-thread sizer's capped
// path would corrupt reverse-mode gradients (the captured `MaxOverRange`'s 1<<24-truncated result undersizes the
// heap), so failing loud is strictly safer. Quadrants's official Vulkan target is `VK_API_VERSION_1_3`, which
// promotes both `VK_KHR_buffer_device_address` and `VK_KHR_shader_atomic_int64` into core; Metal's
// `MTLArgumentBuffersTier::Tier2` (macOS 11+) advertises both caps too. The empty-return branch is forward-looking.
std::vector<uint32_t> build_adstack_max_reducer_spirv(Arch arch, const DeviceCapabilityConfig *caps);

// Compute-shader workgroup size (x dimension; y and z are 1). Power-of-two and a multiple of typical subgroup widths
// on Metal / Vulkan so the workgroup-shared-memory reduction tree contracts at full subgroup width on every step.
// Host launcher uses this to compute `num_workgroups_x = (length + kAdStackMaxReducerWorkgroupSize - 1) /
// kAdStackMaxReducerWorkgroupSize` per dispatch.
constexpr uint32_t kAdStackMaxReducerWorkgroupSize = 128;

// Maximum number of `AdStackSizeExprDeviceNode`s a single spec's body bytecode may contain. The shader's per-thread
// post-order interpreter stores per-node i64 values in a Function-scope array sized by this constant; bumping it
// raises the per-thread stack footprint by 8 bytes/node. The Stage 1 grammar bodies observed on the canonical Genesis
// repro have 3-5 nodes; setting the cap at 64 leaves several orders of magnitude of headroom while keeping the
// per-thread stack at 512 bytes (well below Metal's 4 KiB per-invocation private-memory budget). The host encoder
// hard-errors when a body subtree exceeds this cap so the shader's array bounds are statically known.
constexpr uint32_t kAdStackMaxReducerMaxBodyNodes = 64;

// Layout of the parameter blob the host writes into binding 2 before each dispatch. POD; keep field order in sync
// with the shader's compile-time word-offset constants in `adstack_max_reducer_shader.cpp`. Fields not relevant to a
// particular spec (e.g. `begin_hi` when `begin` fits in 32 bits) are zero-initialised by the host launcher.
struct AdStackMaxReducerParams {
  // Slot index in the per-kernel `BufferType::AdStackMaxReducerOutput` u64 array that this dispatch's atomic-maxes
  // accumulate into. Keyed by `(registry_id, stack_id, mor_node_idx)` packed into a single u32 by the host launcher's
  // `MaxReducerCacheKey -> output_slot` allocator (mirrors how `AdStackBoundReducerParams::task_id_in_kernel` is
  // assigned to `BufferType::AdStackRowCounter` slots).
  uint32_t output_slot;
  // Number of threads to dispatch over (`end - begin` of the captured `MaxOverRange`, evaluated host-side against
  // the live ctx by `evaluate_adstack_size_expr` over the closed-form `begin` and `end` subtrees). Threads with
  // `gl_GlobalInvocationID.x >= length` early-return so dispatch can be rounded up to the workgroup-size multiple
  // without overcounting.
  uint32_t length;
  // Low 32 bits of the captured `MaxOverRange.begin` value, evaluated host-side. The shader reassembles
  // `(begin_hi << 32) | begin_lo` and uses `iter_var = gid + begin` as the body's bound-variable register.
  uint32_t begin_lo;
  // High 32 bits of `begin`. Zero for the typical `MaxOverRange(0, ..., body)` shape.
  uint32_t begin_hi;
  // u32 word offset into the shared bytecode buffer (binding 3) where this spec's body bytecode begins. The bytecode
  // is laid out as `kAdStackSizeExprDeviceMaxBoundVars`-renumbered `AdStackSizeExprDeviceNode`s in post-order, plus a
  // trailing index-entry table at offset `body_bytecode_offset_words + body_node_count * kNodeWords`. The shader
  // walks ascending node indices `0..body_node_count` reading nodes at `body_bytecode_offset_words + i * kNodeWords`.
  uint32_t body_bytecode_offset_words;
  // Number of nodes in this spec's body bytecode. Must satisfy `body_node_count <= kAdStackMaxReducerMaxBodyNodes`;
  // the host encoder checks this and routes the spec back to the capped fallback path if exceeded.
  uint32_t body_node_count;
  // u32 word offset within the shared bytecode buffer where this spec's index-entry table begins (i.e.
  // `body_bytecode_offset_words + body_node_count * kNodeWordsPerNode`). Cached here rather than recomputed in the
  // shader so the shader can index the table without a multiply.
  uint32_t body_indices_offset_words;
  // u32 word offset within the shared bytecode buffer where this spec's `AdStackMaxReducerNode::const_value` payload
  // table begins (one i64 per node, parallel to the node table). Cached for the same reason as above. Empty / unused
  // when `body_node_count == 0`.
  uint32_t body_const_values_offset_words;

  // Offset into the parameter blob (in u32 words) for each field; published to the shader and the host launcher as
  // compile-time constants so each side reads/writes the same slots without a separate header serialisation step.
  static constexpr uint32_t kWordOffsetOutputSlot = 0;
  static constexpr uint32_t kWordOffsetLength = 1;
  static constexpr uint32_t kWordOffsetBeginLo = 2;
  static constexpr uint32_t kWordOffsetBeginHi = 3;
  static constexpr uint32_t kWordOffsetBodyBytecodeOffsetWords = 4;
  static constexpr uint32_t kWordOffsetBodyNodeCount = 5;
  static constexpr uint32_t kWordOffsetBodyIndicesOffsetWords = 6;
  static constexpr uint32_t kWordOffsetBodyConstValuesOffsetWords = 7;
  static constexpr uint32_t kNumWords = 8;
};

}  // namespace quadrants::lang::spirv
