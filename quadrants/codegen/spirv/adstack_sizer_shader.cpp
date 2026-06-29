#include "quadrants/common/logging.h"

#include "quadrants/codegen/spirv/adstack_sizer_shader.h"

#include <array>
#include <string>

#include "quadrants/codegen/spirv/spirv_ir_builder.h"
#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/ir/type.h"

namespace quadrants::lang::spirv {

namespace {

// Capacity limits for the iterative stack machine. Sized generously against observed reverse-mode kernels (max
// tree depth observed ~10, max distinct bound variables ~4) so the shader can cover every realistic case
// without per-launch specialisation. Hitting any of these caps at runtime would indicate a pre-pass that
// produces an unusually deep or wide tree; the host-side encoder hard-errors rather than let the shader
// silently truncate (see `encode_adstack_size_expr_device_bytecode_for_spirv`), so exceeding a cap surfaces
// as a clear compile-time diagnostic rather than a mysterious overflow at the next `stack_push`.
// Per-stack node-count cap for the eval state. The shader hosts its `values_arr` slice in the i64 scratch
// SSBO (binding 3) sized at `kAdStackSizerScratchI64Elems`; the slice is indexed by the *local* offset
// within the stack's subtree (not the global bytecode-wide node index), so this cap applies only per-stack,
// not per-kernel. Observed reverse-mode kernels have a handful of stacks with ~1k-node symbolic trees
// (`MaxOverRange` over ndarray reads with a nested `Max` of per-substep trip counts); 65536 * 8 B = 512 KiB
// of device memory per `GfxRuntime` (allocated once, reused across every sizer dispatch).
constexpr int kMaxNodes = kAdStackSizerMaxNodesPerStack;
// `kMaxVars` sizes the per-invocation `scope_arr` indexed by dense bound-var id; must match
// `kAdStackSizeExprDeviceMaxBoundVars` so every id the host-side dense-remap hands out lands in bounds. `kMaxPending`
// bounds `MaxOverRange` nesting depth; the host encoder in `encode_bytecode_common` hard-errors when a tree's MOR
// nesting exceeds this, so the shader's OpAccessChain into `pending_*_arr[sp_val]` is always in range.
constexpr int kMaxVars = kAdStackSizeExprDeviceMaxBoundVars;
constexpr int kMaxPending = kAdStackSizerMaxPendingFrames;

// Byte-per-u32 counts derived from the POD layout in `quadrants/ir/adstack_size_expr_device.h`. Every field
// in the device structs is 4-byte aligned, so we can index into a `uint32_t[]` view of the bytecode buffer
// without worrying about alignment. Doubled up for `int64` by reading two adjacent u32 slots.
constexpr int kHeaderWords = sizeof(AdStackSizeExprDeviceHeader) / sizeof(uint32_t);
constexpr int kStackHeaderWords = sizeof(AdStackSizeExprDeviceStackHeader) / sizeof(uint32_t);
constexpr int kNodeWords = sizeof(AdStackSizeExprDeviceNode) / sizeof(uint32_t);
static_assert(kHeaderWords == 4, "device header layout changed; shader word offsets must be updated");
static_assert(kStackHeaderWords == 4, "device stack-header layout changed; shader word offsets must be updated");
static_assert(kNodeWords == 12, "device node layout changed; shader word offsets must be updated");

// Word offsets of each field within a `AdStackSizeExprDeviceNode` when viewed as `uint32_t[12]`. Kept in one
// place here rather than scattered through the SPIR-V emission so a future layout change is a single-file
// fix.
constexpr int kNodeOffKind = 0;
constexpr int kNodeOffOperandA = 1;
constexpr int kNodeOffOperandB = 2;
constexpr int kNodeOffBodyNodeIdx = 3;
constexpr int kNodeOffVarId = 4;
constexpr int kNodeOffPrimDt = 5;
constexpr int kNodeOffArgBufferOffset = 6;
constexpr int kNodeOffIndicesOffset = 7;
constexpr int kNodeOffIndicesCount = 8;
// slot 9 is `_pad0`
constexpr int kNodeOffConstLo =
    10;  // `const_value` is i64: little-endian lo word then hi word consumed by `load_buf_i64`

// Word offsets within a `AdStackSizeExprDeviceStackHeader` when viewed as `uint32_t[4]`. `entry_size_bytes`
// at word 1 is unused by the SPIR-V shader (the running-offset math multiplies by 2 for float / 1 for int
// in elements, not bytes, per the convention baked into `encode_adstack_size_expr_device_bytecode_for_spirv`);
// omit the constant so `-Werror -Wunused-const-variable` stays clean.
constexpr int kStackOffRootNodeIdx = 0;
constexpr int kStackOffMaxSizeCompileTime = 2;
constexpr int kStackOffHeapKind = 3;

// Word offsets within `AdStackSizeExprDeviceHeader` when viewed as `uint32_t[4]`.
constexpr int kHeaderOffNStacks = 0;
constexpr int kHeaderOffTotalNodes = 1;

// `PrimitiveTypeID` numeric values (mirrored from `quadrants/inc/data_type.inc.h` so the shader doesn't need
// to include that via the Quadrants `Type` header). The enum order is fixed by the `PER_TYPE` macro sequence
// in that file and a static_assert would require pulling in heavy headers for the sake of one cross-check;
// the comment is the invariant documentation instead.
//   f16=0, f32=1, f64=2, i8=3, i16=4, i32=5, i64=6, u1=7, u8=8, u16=9, u32=10, u64=11
constexpr int kPrimI8 = 3;
constexpr int kPrimI16 = 4;
constexpr int kPrimI32 = 5;
constexpr int kPrimI64 = 6;
constexpr int kPrimU8 = 8;
constexpr int kPrimU16 = 9;
constexpr int kPrimU32 = 10;
constexpr int kPrimU64 = 11;

// Small helper: read one uint32 word from a storage-buffer-backed uint32[] at the given scalar index.
Value load_buf_u32(IRBuilder &ir, Value buffer, Value word_idx) {
  Value ptr = ir.struct_array_access(ir.u32_type(), buffer, word_idx);
  return ir.load_variable(ptr, ir.u32_type());
}

// Helper: assemble an i64 from two adjacent little-endian u32 words in a buffer. Assumes the buffer is
// logically uint32[] - correct for both the bytecode buffer (encoder memcpys POD structs in their native
// layout, which is little-endian on every platform Quadrants supports) and the kernel arg buffer (same).
Value load_buf_i64(IRBuilder &ir, Value buffer, Value base_word_idx) {
  Value lo = load_buf_u32(ir, buffer, base_word_idx);
  Value hi_idx = ir.add(base_word_idx, ir.uint_immediate_number(ir.u32_type(), 1u));
  Value hi = load_buf_u32(ir, buffer, hi_idx);
  // Zero-extend both halves to u64, shift hi, OR them, reinterpret as i64.
  Value lo64 = ir.cast(ir.u64_type(), lo);
  Value hi64 = ir.cast(ir.u64_type(), hi);
  Value shift = ir.uint_immediate_number(ir.u64_type(), 32u);
  Value hi_shifted = ir.make_value(spv::OpShiftLeftLogical, ir.u64_type(), hi64, shift);
  Value or_val = ir.make_value(spv::OpBitwiseOr, ir.u64_type(), lo64, hi_shifted);
  // Reinterpret as i64 via OpBitcast.
  return ir.make_value(spv::OpBitcast, ir.i64_type(), or_val);
}

// Helper: write a u32 word into a storage-buffer-backed uint32[].
void store_buf_u32(IRBuilder &ir, Value buffer, Value word_idx, Value value) {
  Value ptr = ir.struct_array_access(ir.u32_type(), buffer, word_idx);
  ir.store_variable(ptr, value);
}

// Access an i64 / i32 slot inside the per-invocation interpreter scratch state. The state is hosted in two
// `StorageBuffer` SSBOs (binding 3 = i64 scratch, binding 4 = i32 scratch) rather than `Function`-storage arrays
// because Blackwell-class NVIDIA Vulkan drivers cap per-thread private memory at a budget the cumulative
// `i64[kMaxNodes] + i64[kMaxVars] + 8x array[kMaxPending]` (~34 KiB) blows through, failing `vkCreateComputePipelines`
// with `VK_ERROR_UNKNOWN`. Each per-array slice within a scratch SSBO has a fixed compile-time base index
// (`base_in_elems`); element access is `base_in_elems + index` then `OpAccessChain` through the SSBO's runtime array.
// The sizer dispatches `1x1x1` so cross-thread aliasing is impossible.
Value array_i64_access_ptr(IRBuilder &ir, Value scratch_i64_buf, uint32_t base_in_elems, Value index) {
  Value abs_idx = ir.add(ir.uint_immediate_number(ir.u32_type(), base_in_elems), ir.cast(ir.u32_type(), index));
  return ir.struct_array_access(ir.i64_type(), scratch_i64_buf, abs_idx);
}

Value array_i32_access_ptr(IRBuilder &ir, Value scratch_i32_buf, uint32_t base_in_elems, Value index) {
  Value abs_idx = ir.add(ir.uint_immediate_number(ir.u32_type(), base_in_elems), ir.cast(ir.u32_type(), index));
  return ir.struct_array_access(ir.i32_type(), scratch_i32_buf, abs_idx);
}

// PSB load of one scalar from a physical-storage-buffer-addressed pointer. `base_u64` is the pointer value
// (already as a u64 SSA value); `elem_idx_i32` is the scalar element index (i32 is fine for the index range
// we care about - ndarray shapes are bounded well under `INT32_MAX`); `elem_sty` is the SPIR-V type to load.
// Computes the target address as `base_u64 + elem_idx * sizeof(elem)` and does `OpConvertUToPtr` directly to
// a pointer-to-wrapper-struct, then reads the scalar via `OpAccessChain` on the `_m0` member. This bypasses
// `OpPtrAccessChain`, which requires an `ArrayStride` decoration on the pointer / pointee type that we
// don't have a clean entry point to add here (the main codegen only decorates `OpTypeArray` - see
// `get_array_type`). The wrapper-struct approach is also the fallback path `TaskCodegen::at_buffer` uses
// when `physical_ptr_components_` doesn't carry a decomposed base+index pair, so we're staying within a
// pattern SPIRV-Cross already knows how to lower on Metal / Vulkan without tripping the
// rvalue-pointer-to-atomic MSL miscompile.
Value psb_load_scalar(IRBuilder &ir,
                      Value base_u64,
                      Value elem_idx_i32,
                      const SType &elem_sty,
                      size_t elem_size_bytes) {
  Value elem_size_u64 = ir.uint_immediate_number(ir.u64_type(), elem_size_bytes);
  Value elem_idx_u64 = ir.cast(ir.u64_type(), elem_idx_i32);
  Value byte_off = ir.mul(elem_idx_u64, elem_size_u64);
  Value target_u64 = ir.add(base_u64, byte_off);

  SType ptr_elem_type = ir.get_pointer_type(elem_sty, spv::StorageClassPhysicalStorageBuffer);
  std::vector<std::tuple<SType, std::string, size_t>> members = {{elem_sty, "_m0", 0}};
  SType wrapper_struct = ir.create_struct_type(members);
  SType ptr_struct_type = ir.get_pointer_type(wrapper_struct, spv::StorageClassPhysicalStorageBuffer);
  Value struct_ptr = ir.make_value(spv::OpConvertUToPtr, ptr_struct_type, target_u64);
  Value scalar_ptr = ir.make_value(spv::OpAccessChain, ptr_elem_type, struct_ptr, ir.const_i32_zero_);
  // `OpLoad` through a `PhysicalStorageBuffer` pointer requires the `Aligned` memory-access operand per the
  // SPIR-V spec; without it `spirv-val` errors with "Memory accesses with PhysicalStorageBuffer must use
  // Aligned". The alignment equals the element size for scalar loads, since the encoder's byte-offset
  // arithmetic above preserves natural alignment (all ndarray element types Quadrants supports are
  // power-of-two sizes up to 8 bytes).
  Value scalar = ir.new_value(elem_sty, ValueKind::kNormal);
  ir.make_inst(spv::OpLoad, elem_sty, scalar, scalar_ptr, spv::MemoryAccessAlignedMask,
               static_cast<uint32_t>(elem_size_bytes));
  return scalar;
}

// Emit the per-kind dispatch block. Each case loads operands from `values_arr` (or scope), computes the
// result i64, and stores back to `values_arr[current]`. Kinds that alter control flow (`MaxOverRange`) do
// their own branches and return early from the dispatch by branching to the enclosing merge label.
//
// Takes all the state variables so it can read/write them directly. Returns `std::nullopt`-equivalent by
// always branching to `after_dispatch_label` (the kind-switch merge point). The caller sets up the switch
// surround.
// Compile-time element offsets into the i64 scratch SSBO. Layout: values_arr | scope_arr | pending_cur_i |
// pending_end | pending_max_accum. Total element count is published as `kAdStackSizerScratchI64Elems` in
// `adstack_sizer_shader.h` for the host launcher to size the binding-3 buffer.
constexpr uint32_t kI64BaseValuesArr = 0;
constexpr uint32_t kI64BaseScopeArr = kMaxNodes;
constexpr uint32_t kI64BasePendingCurI = kMaxNodes + kMaxVars;
constexpr uint32_t kI64BasePendingEnd = kMaxNodes + kMaxVars + kMaxPending;
constexpr uint32_t kI64BasePendingMaxAccum = kMaxNodes + kMaxVars + 2 * kMaxPending;

// Compile-time element offsets into the i32 scratch SSBO. Layout: pending_mor_idx | pending_body_start |
// pending_body_end | pending_var_id | pending_saved_max_k. Total element count is published as
// `kAdStackSizerScratchI32Elems` in `adstack_sizer_shader.h`.
constexpr uint32_t kI32BasePendingMorIdx = 0;
constexpr uint32_t kI32BasePendingBodyStart = kMaxPending;
constexpr uint32_t kI32BasePendingBodyEnd = 2 * kMaxPending;
constexpr uint32_t kI32BasePendingVarId = 3 * kMaxPending;
constexpr uint32_t kI32BasePendingSavedMaxK = 4 * kMaxPending;
static_assert(kI64BasePendingMaxAccum + kMaxPending == static_cast<uint32_t>(kAdStackSizerScratchI64Elems),
              "i64 scratch layout drifted from header element count");
static_assert(kI32BasePendingSavedMaxK + kMaxPending == static_cast<uint32_t>(kAdStackSizerScratchI32Elems),
              "i32 scratch layout drifted from header element count");

struct ShaderState {
  // Scratch SSBOs hosting the interpreter state. The per-array slices below live at compile-time element
  // offsets into these buffers; see `kI64Base*` / `kI32Base*` above.
  Value scratch_i64_buf;
  Value scratch_i32_buf;
  // Per-thread scalar variables. These are tiny (a handful of u32 / i32) so Function storage is fine -
  // Blackwell's per-thread-private cap only kicked in for the cumulative ~34 KiB of array storage that now
  // lives in the scratch SSBOs.
  Value current_var;            // alloca i32
  Value max_k_var;              // alloca i32
  Value sp_var;                 // alloca i32
  Value nodes_base_word_var;    // alloca u32 (base word offset of nodes array)
  Value indices_base_word_var;  // alloca u32
  Value tree_start_var;         // alloca i32, global node index of the current stack's tree root-walk origin
  Value bytecode_buf;
  Value metadata_buf;
  Value args_buf;
  // Word index inside `metadata_buf` of the trailing overflow-flag slot. Computed once per dispatch in `main` (`2 + 2 *
  // n_stacks`); the per-stack walker writes 1 here when it observes a `MaxOverRange` whose iteration count exceeds the
  // `1<<24` cap. The host launcher's post-readback path raises a `QuadrantsAssertionError` when the slot is non-zero,
  // so the cap-hit surfaces as a clean error rather than an under-bounded heap.
  Value overflow_flag_word_var;
};

// `values_arr` is a private, function-local i64 array of size `kMaxNodes` used to memoise the value of every
// node in the *current stack's* tree as the walker processes it in post-order. Indexing it by the node's
// global post-order position (i.e. directly by `current_now`) only works when the whole kernel has <=
// kMaxNodes nodes in total, which fails for large reverse-mode kernels (upwards of 20k total nodes across 700+
// stacks). We index by the per-stack local offset instead: `values_arr[current_now - tree_start]`, where
// `tree_start` is the global index of this stack's first node. Every node in a single stack's subtree is
// reachable within the [tree_start, root_idx] range and the kMaxNodes cap therefore applies only per-stack,
// not per-kernel. All operand_a / operand_b / body_node_idx values are already in the same global frame so
// subtracting `tree_start` keeps the addressing consistent across kinds.
Value local_values_idx(IRBuilder &ir, const ShaderState &st, Value global_index_i32) {
  Value tree_start = ir.load_variable(st.tree_start_var, ir.i32_type());
  return ir.sub(global_index_i32, tree_start);
}

Value load_values_at(IRBuilder &ir, const ShaderState &st, Value index_i32) {
  Value ptr = array_i64_access_ptr(ir, st.scratch_i64_buf, kI64BaseValuesArr, local_values_idx(ir, st, index_i32));
  return ir.load_variable(ptr, ir.i64_type());
}

void store_values_at(IRBuilder &ir, const ShaderState &st, Value index_i32, Value v_i64) {
  Value ptr = array_i64_access_ptr(ir, st.scratch_i64_buf, kI64BaseValuesArr, local_values_idx(ir, st, index_i32));
  ir.store_variable(ptr, v_i64);
}

Value load_scope_at(IRBuilder &ir, const ShaderState &st, Value var_id_i32) {
  Value ptr = array_i64_access_ptr(ir, st.scratch_i64_buf, kI64BaseScopeArr, var_id_i32);
  return ir.load_variable(ptr, ir.i64_type());
}

void store_scope_at(IRBuilder &ir, const ShaderState &st, Value var_id_i32, Value v_i64) {
  Value ptr = array_i64_access_ptr(ir, st.scratch_i64_buf, kI64BaseScopeArr, var_id_i32);
  ir.store_variable(ptr, v_i64);
}

// Compute the u32 word offset of field `word_off` inside the device node at index `node_idx`.
Value node_field_word_idx(IRBuilder &ir, Value nodes_base_word, Value node_idx_i32, int word_off_in_node) {
  Value node_idx_u32 = ir.cast(ir.u32_type(), node_idx_i32);
  Value node_words = ir.uint_immediate_number(ir.u32_type(), kNodeWords);
  Value base = ir.mul(node_idx_u32, node_words);
  Value base_plus = ir.add(nodes_base_word, base);
  Value off = ir.uint_immediate_number(ir.u32_type(), static_cast<uint32_t>(word_off_in_node));
  return ir.add(base_plus, off);
}

// `kExternalTensorRead` shares the pair-based indices layout `[idx_a_raw, elem_stride_a]` with `kFieldLoad`
// after the host encoder change: the only difference between the two device-node kinds is where the base
// pointer lives (arg-buffer slot vs snode PSB base stashed in `const_value`). Both use
// `compute_field_load_elem_index` below to walk the indices and accumulate `v * elem_stride` per axis;
// there is no longer a stride-1 ETR helper because that path silently dropped multi-axis ndarrays onto the
// wrong flat element and tripped an `Adstack overflow` at `qd.sync()`.

// FieldLoad's indices layout is `[idx_0_raw, elem_stride_0, idx_1_raw, elem_stride_1, ...]`, so `indices_count`
// counts axes and the actual buffer range is `indices[off .. off + 2 * count)`. Each pair contributes
// `idx_a * stride_a` to the element index. Mirrors `compute_linear_index` for ETR but walks pairs and multiplies
// in the per-axis stride read from the second half of each pair, matching how the host encoder emits the table.
Value compute_field_load_elem_index(IRBuilder &ir,
                                    const ShaderState &st,
                                    Value indices_base_word,  // u32
                                    Value indices_offset_i32,
                                    Value indices_count_i32) {
  // Element-index accumulator. Sized to i32 like the ETR path - observed snode shapes are well below 2^31.
  Value acc_var = ir.alloca_variable(ir.i32_type());
  ir.store_variable(acc_var, ir.int_immediate_number(ir.i32_type(), 0));

  Value k_var = ir.alloca_variable(ir.i32_type());
  ir.store_variable(k_var, ir.int_immediate_number(ir.i32_type(), 0));

  Label head = ir.new_label();
  Label body = ir.new_label();
  Label cont = ir.new_label();
  Label merge = ir.new_label();

  ir.make_inst(spv::OpBranch, head);

  ir.start_label(head);
  Value k_now = ir.load_variable(k_var, ir.i32_type());
  Value cond = ir.lt(k_now, indices_count_i32);
  ir.make_inst(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
  ir.make_inst(spv::OpBranchConditional, cond, body, merge);

  ir.start_label(body);
  Value indices_off_u32 = ir.cast(ir.u32_type(), indices_offset_i32);
  // Pair offset for axis k = off + 2*k (idx at +0, elem_stride at +1).
  Value k_u32 = ir.cast(ir.u32_type(), k_now);
  Value two_u32 = ir.uint_immediate_number(ir.u32_type(), 2u);
  Value pair_base_u32 = ir.add(indices_off_u32, ir.mul(k_u32, two_u32));
  Value idx_word_u32 = ir.add(indices_base_word, pair_base_u32);
  Value stride_word_u32 = ir.add(idx_word_u32, ir.uint_immediate_number(ir.u32_type(), 1u));
  Value idx_raw_u32 = load_buf_u32(ir, st.bytecode_buf, idx_word_u32);
  Value idx_raw_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), idx_raw_u32);
  Value stride_u32 = load_buf_u32(ir, st.bytecode_buf, stride_word_u32);
  Value stride_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), stride_u32);

  // v = raw >= 0 ? raw : scope[-(raw+1)]  ; then acc += v * stride.
  Label const_lbl = ir.new_label();
  Label var_lbl = ir.new_label();
  Label sel_merge = ir.new_label();
  Value is_const = ir.ge(idx_raw_i32, ir.int_immediate_number(ir.i32_type(), 0));
  ir.make_inst(spv::OpSelectionMerge, sel_merge, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpBranchConditional, is_const, const_lbl, var_lbl);

  ir.start_label(const_lbl);
  Value v_const_i32 = idx_raw_i32;
  ir.make_inst(spv::OpBranch, sel_merge);

  ir.start_label(var_lbl);
  Value raw_plus_1 = ir.add(idx_raw_i32, ir.int_immediate_number(ir.i32_type(), 1));
  Value var_id_i32 = ir.sub(ir.int_immediate_number(ir.i32_type(), 0), raw_plus_1);
  Value scope_val_i64 = load_scope_at(ir, st, var_id_i32);
  Value v_var_i32 = ir.cast(ir.i32_type(), scope_val_i64);
  Label var_lbl_end = ir.current_label();
  ir.make_inst(spv::OpBranch, sel_merge);

  ir.start_label(sel_merge);
  PhiValue v = ir.make_phi(ir.i32_type(), 2);
  v.set_incoming(0, v_const_i32, const_lbl);
  v.set_incoming(1, v_var_i32, var_lbl_end);

  Value contribution = ir.mul(Value(v), stride_i32);
  Value acc_now = ir.load_variable(acc_var, ir.i32_type());
  Value acc_next = ir.add(acc_now, contribution);
  ir.store_variable(acc_var, acc_next);
  ir.make_inst(spv::OpBranch, cont);

  ir.start_label(cont);
  Value k_next = ir.add(k_now, ir.int_immediate_number(ir.i32_type(), 1));
  ir.store_variable(k_var, k_next);
  ir.make_inst(spv::OpBranch, head);

  ir.start_label(merge);
  return ir.load_variable(acc_var, ir.i32_type());
}

// Returns `sizeof(prim_dt)` as an i64 SSA value, matching the primitive type the `kFieldLoad` / `kExternalTensorRead`
// shader switch uses. Kept in one place so the element-size-of-prim-dt table stays in sync with `emit_psb_load_i64`.
// Emit the switch-on-prim_dt + PSB load sequence. Returns an i64.
Value emit_psb_load_i64(IRBuilder &ir, Value data_ptr_u64, Value linear_i32, Value prim_dt_i32) {
  // Use nested if-else tree (SPIR-V has OpSwitch too, but OpSwitch on integer types needs a literal case
  // list, which SPIRV-Cross handles fine - we use it here for compactness).
  Label merge = ir.new_label();
  Label case_i8 = ir.new_label();
  Label case_i16 = ir.new_label();
  Label case_i32 = ir.new_label();
  Label case_i64 = ir.new_label();
  Label case_u8 = ir.new_label();
  Label case_u16 = ir.new_label();
  Label case_u32 = ir.new_label();
  Label case_u64 = ir.new_label();
  Label case_default = ir.new_label();

  ir.make_inst(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpSwitch, prim_dt_i32, case_default, static_cast<uint32_t>(kPrimI8), case_i8,
               static_cast<uint32_t>(kPrimI16), case_i16, static_cast<uint32_t>(kPrimI32), case_i32,
               static_cast<uint32_t>(kPrimI64), case_i64, static_cast<uint32_t>(kPrimU8), case_u8,
               static_cast<uint32_t>(kPrimU16), case_u16, static_cast<uint32_t>(kPrimU32), case_u32,
               static_cast<uint32_t>(kPrimU64), case_u64);

  auto emit_case = [&](Label lbl, const SType &load_ty, size_t elem_size, bool is_signed) -> std::pair<Value, Label> {
    ir.start_label(lbl);
    Value v = psb_load_scalar(ir, data_ptr_u64, linear_i32, load_ty, elem_size);
    // Sign-extend (signed) or zero-extend (unsigned) to i64.
    Value v_i64;
    if (is_signed) {
      v_i64 = ir.make_value(spv::OpSConvert, ir.i64_type(), v);
    } else {
      // Widen unsigned to u64, reinterpret to i64.
      Value v_u64 = ir.make_value(spv::OpUConvert, ir.u64_type(), v);
      v_i64 = ir.make_value(spv::OpBitcast, ir.i64_type(), v_u64);
    }
    Label out = ir.current_label();
    ir.make_inst(spv::OpBranch, merge);
    return {v_i64, out};
  };

  auto [v_i8, lbl_i8] = emit_case(case_i8, ir.i8_type(), 1, true);
  auto [v_i16, lbl_i16] = emit_case(case_i16, ir.i16_type(), 2, true);
  auto [v_i32, lbl_i32] = emit_case(case_i32, ir.i32_type(), 4, true);
  // i64 is the widest - OpSConvert i64->i64 is invalid, so just rebind.
  ir.start_label(case_i64);
  Value v_i64_direct = psb_load_scalar(ir, data_ptr_u64, linear_i32, ir.i64_type(), 8);
  Label lbl_i64 = ir.current_label();
  ir.make_inst(spv::OpBranch, merge);

  auto [v_u8, lbl_u8] = emit_case(case_u8, ir.u8_type(), 1, false);
  auto [v_u16, lbl_u16] = emit_case(case_u16, ir.u16_type(), 2, false);
  auto [v_u32, lbl_u32] = emit_case(case_u32, ir.u32_type(), 4, false);
  ir.start_label(case_u64);
  Value v_u64_raw = psb_load_scalar(ir, data_ptr_u64, linear_i32, ir.u64_type(), 8);
  Value v_u64_as_i64 = ir.make_value(spv::OpBitcast, ir.i64_type(), v_u64_raw);
  Label lbl_u64 = ir.current_label();
  ir.make_inst(spv::OpBranch, merge);

  // Default: return 0.
  ir.start_label(case_default);
  Value zero_i64 = ir.int_immediate_number(ir.i64_type(), 0);
  Label lbl_default = ir.current_label();
  ir.make_inst(spv::OpBranch, merge);

  ir.start_label(merge);
  PhiValue result = ir.make_phi(ir.i64_type(), 9);
  result.set_incoming(0, v_i8, lbl_i8);
  result.set_incoming(1, v_i16, lbl_i16);
  result.set_incoming(2, v_i32, lbl_i32);
  result.set_incoming(3, v_i64_direct, lbl_i64);
  result.set_incoming(4, v_u8, lbl_u8);
  result.set_incoming(5, v_u16, lbl_u16);
  result.set_incoming(6, v_u32, lbl_u32);
  result.set_incoming(7, v_u64_as_i64, lbl_u64);
  result.set_incoming(8, zero_i64, lbl_default);
  return Value(result);
}

// Emit the tree-eval inner loop for the current stack. On entry: `current_var`, `max_k_var`, `sp_var` are
// initialised by the caller. On exit: `values_arr[root_idx]` holds the tree's root value. Uses the
// pending-frames stack for `MaxOverRange` iteration. See the top-level module comment for the algorithm.
void emit_tree_eval_loop(IRBuilder &ir, const ShaderState &st) {
  Label head = ir.new_label();
  Label body = ir.new_label();
  Label cont = ir.new_label();
  Label merge = ir.new_label();
  ir.make_inst(spv::OpBranch, head);

  ir.start_label(head);
  ir.make_inst(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
  ir.make_inst(spv::OpBranch, body);

  ir.start_label(body);
  Value current_now = ir.load_variable(st.current_var, ir.i32_type());
  Value max_k_now = ir.load_variable(st.max_k_var, ir.i32_type());
  Value sp_now = ir.load_variable(st.sp_var, ir.i32_type());
  Value past_end = ir.gt(current_now, max_k_now);

  // Branch: if past the end, handle pending frame pop/continuation; else process the node.
  Label past_lbl = ir.new_label();
  Label exec_lbl = ir.new_label();
  Label body_merge = ir.new_label();
  ir.make_inst(spv::OpSelectionMerge, body_merge, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpBranchConditional, past_end, past_lbl, exec_lbl);

  // ---- past-end branch ----
  ir.start_label(past_lbl);
  Value sp_is_zero = ir.eq(sp_now, ir.int_immediate_number(ir.i32_type(), 0));

  Label stop_lbl = ir.new_label();  // sp == 0 -> exit loop
  Label pop_lbl = ir.new_label();   // sp > 0 -> update pending frame
  Label past_merge = ir.new_label();
  ir.make_inst(spv::OpSelectionMerge, past_merge, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpBranchConditional, sp_is_zero, stop_lbl, pop_lbl);

  ir.start_label(stop_lbl);
  ir.make_inst(spv::OpBranch, merge);  // exit the outer loop entirely

  ir.start_label(pop_lbl);
  // top_idx = sp - 1
  Value top_idx = ir.sub(sp_now, ir.int_immediate_number(ir.i32_type(), 1));
  // body_result = values[pending_body_end[top_idx]]
  Value body_end_ptr = array_i32_access_ptr(ir, st.scratch_i32_buf, kI32BasePendingBodyEnd, top_idx);
  Value body_end_node = ir.load_variable(body_end_ptr, ir.i32_type());
  Value body_result = load_values_at(ir, st, body_end_node);
  // top_max = max(pending_max_accum[top_idx], body_result)
  Value max_accum_ptr = array_i64_access_ptr(ir, st.scratch_i64_buf, kI64BasePendingMaxAccum, top_idx);
  Value cur_max_accum = ir.load_variable(max_accum_ptr, ir.i64_type());
  Value body_gt_accum = ir.gt(body_result, cur_max_accum);
  Value new_max_accum = ir.select(body_gt_accum, body_result, cur_max_accum);
  ir.store_variable(max_accum_ptr, new_max_accum);
  // top_cur_i = pending_cur_i[top_idx] + 1
  Value cur_i_ptr = array_i64_access_ptr(ir, st.scratch_i64_buf, kI64BasePendingCurI, top_idx);
  Value cur_i = ir.load_variable(cur_i_ptr, ir.i64_type());
  Value next_i = ir.add(cur_i, ir.int_immediate_number(ir.i64_type(), 1));
  // if next_i < pending_end[top_idx]: update cur_i, jump to body_start
  Value end_ptr = array_i64_access_ptr(ir, st.scratch_i64_buf, kI64BasePendingEnd, top_idx);
  Value end_val = ir.load_variable(end_ptr, ir.i64_type());
  Value more = ir.lt(next_i, end_val);
  Label more_lbl = ir.new_label();
  Label done_lbl = ir.new_label();
  Label past_step_merge = ir.new_label();
  ir.make_inst(spv::OpSelectionMerge, past_step_merge, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpBranchConditional, more, more_lbl, done_lbl);

  ir.start_label(more_lbl);
  ir.store_variable(cur_i_ptr, next_i);
  // scope[pending_var_id[top_idx]] = next_i
  Value var_id_ptr = array_i32_access_ptr(ir, st.scratch_i32_buf, kI32BasePendingVarId, top_idx);
  Value var_id = ir.load_variable(var_id_ptr, ir.i32_type());
  store_scope_at(ir, st, var_id, next_i);
  // current = pending_body_start[top_idx]
  Value body_start_ptr = array_i32_access_ptr(ir, st.scratch_i32_buf, kI32BasePendingBodyStart, top_idx);
  Value body_start_val = ir.load_variable(body_start_ptr, ir.i32_type());
  ir.store_variable(st.current_var, body_start_val);
  // max_k stays (we're still iterating the same MOR body)
  ir.make_inst(spv::OpBranch, past_step_merge);

  ir.start_label(done_lbl);
  // Clear `scope[pending_var_id]` to zero before popping the frame. The outer `scope_arr` zero-init runs
  // once at `main()` entry, but `var_id_counter` in `compute_bounded_adstack_size` resets per alloca, so
  // different stacks in the same task reuse `scope[0]` / `scope[1]` / ... After this MOR completes the
  // slot holds the last bound value (e.g. `N - 1` for a `[0, N)` range); the NEXT stack's outer linear
  // pre-order walk - which crosses the body subtree of every MaxOverRange once BEFORE the MOR binds the
  // variable - would read that stale value as an index into a potentially-smaller ndarray, triggering an
  // OOB PSB load (Metal: hung command buffer; Vulkan with robustBufferAccess: silent zero feeding a later
  // `Adstack overflow`). Zeroing here preserves the "scope[var_id] == 0 is a safe spurious-read target
  // because index 0 is always valid for any non-empty ndarray" invariant the outer walk relies on.
  Value pop_var_id_ptr = array_i32_access_ptr(ir, st.scratch_i32_buf, kI32BasePendingVarId, top_idx);
  Value pop_var_id = ir.load_variable(pop_var_id_ptr, ir.i32_type());
  Value pop_scope_ptr = array_i64_access_ptr(ir, st.scratch_i64_buf, kI64BaseScopeArr, pop_var_id);
  ir.store_variable(pop_scope_ptr, ir.int_immediate_number(ir.i64_type(), 0));

  // values[pending_mor_idx[top_idx]] = new_max_accum
  Value mor_idx_ptr = array_i32_access_ptr(ir, st.scratch_i32_buf, kI32BasePendingMorIdx, top_idx);
  Value mor_idx = ir.load_variable(mor_idx_ptr, ir.i32_type());
  store_values_at(ir, st, mor_idx, new_max_accum);
  // current = pending_mor_idx[top_idx] + 1. The encoder emits `MaxOverRange` in post-order, i.e. body nodes
  // come BEFORE the MOR node itself (the MOR is the subtree root). Setting `current = body_end + 1` would
  // land right back on the MOR node and re-enter it every pop, producing an unbounded loop on the first
  // reverse-mode kernel with an `ExternalTensorRead`-bounded MaxOverRange. Advance past the MOR instead so
  // the outer linear walk picks up with whatever sibling / parent node follows it in post-order.
  Value cur_next = ir.add(mor_idx, ir.int_immediate_number(ir.i32_type(), 1));
  ir.store_variable(st.current_var, cur_next);
  // max_k = pending_saved_max_k[top_idx]
  Value saved_max_k_ptr = array_i32_access_ptr(ir, st.scratch_i32_buf, kI32BasePendingSavedMaxK, top_idx);
  Value saved_max_k = ir.load_variable(saved_max_k_ptr, ir.i32_type());
  ir.store_variable(st.max_k_var, saved_max_k);
  // sp -= 1
  ir.store_variable(st.sp_var, top_idx);
  ir.make_inst(spv::OpBranch, past_step_merge);

  ir.start_label(past_step_merge);
  ir.make_inst(spv::OpBranch, past_merge);

  ir.start_label(past_merge);
  ir.make_inst(spv::OpBranch, body_merge);

  // ---- execute-node branch ----
  ir.start_label(exec_lbl);
  // Read node header words
  Value nodes_base_word = ir.load_variable(st.nodes_base_word_var, ir.u32_type());
  Value kind_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffKind);
  Value kind_u32 = load_buf_u32(ir, st.bytecode_buf, kind_idx);
  Value kind_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), kind_u32);

  Value op_a_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffOperandA);
  Value op_a_u32 = load_buf_u32(ir, st.bytecode_buf, op_a_idx);
  Value op_a_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), op_a_u32);

  Value op_b_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffOperandB);
  Value op_b_u32 = load_buf_u32(ir, st.bytecode_buf, op_b_idx);
  Value op_b_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), op_b_u32);

  Value body_node_idx_word = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffBodyNodeIdx);
  Value body_node_u32 = load_buf_u32(ir, st.bytecode_buf, body_node_idx_word);
  Value body_node_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), body_node_u32);

  Value var_id_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffVarId);
  Value var_id_u32 = load_buf_u32(ir, st.bytecode_buf, var_id_idx);
  Value var_id_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), var_id_u32);

  // Switch on kind.
  Label case_const = ir.new_label();
  Label case_add = ir.new_label();
  Label case_sub = ir.new_label();
  Label case_mul = ir.new_label();
  Label case_max = ir.new_label();
  Label case_mor = ir.new_label();
  Label case_bv = ir.new_label();
  Label case_etr = ir.new_label();
  Label case_fl = ir.new_label();
  Label case_default = ir.new_label();
  Label exec_merge = ir.new_label();

  ir.make_inst(spv::OpSelectionMerge, exec_merge, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpSwitch, kind_i32, case_default, static_cast<uint32_t>(AdStackSizeExprDeviceKind::kConst),
               case_const, static_cast<uint32_t>(AdStackSizeExprDeviceKind::kAdd), case_add,
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kSub), case_sub,
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kMul), case_mul,
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kMax), case_max,
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kMaxOverRange), case_mor,
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kBoundVariable), case_bv,
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kExternalTensorRead), case_etr,
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kFieldLoad), case_fl);

  auto emit_binary_advance = [&](Label start_lbl, auto compute_fn) {
    ir.start_label(start_lbl);
    Value lhs = load_values_at(ir, st, op_a_i32);
    Value rhs = load_values_at(ir, st, op_b_i32);
    Value v = compute_fn(lhs, rhs);
    store_values_at(ir, st, current_now, v);
    Value next_cur = ir.add(current_now, ir.int_immediate_number(ir.i32_type(), 1));
    ir.store_variable(st.current_var, next_cur);
    ir.make_inst(spv::OpBranch, exec_merge);
  };

  // Const
  ir.start_label(case_const);
  Value const_lo_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffConstLo);
  Value const_val = load_buf_i64(ir, st.bytecode_buf, const_lo_idx);
  store_values_at(ir, st, current_now, const_val);
  Value const_next_cur = ir.add(current_now, ir.int_immediate_number(ir.i32_type(), 1));
  ir.store_variable(st.current_var, const_next_cur);
  ir.make_inst(spv::OpBranch, exec_merge);

  // Add
  emit_binary_advance(case_add, [&](Value a, Value b) { return ir.add(a, b); });
  // Sub (clamped to 0)
  ir.start_label(case_sub);
  {
    Value lhs = load_values_at(ir, st, op_a_i32);
    Value rhs = load_values_at(ir, st, op_b_i32);
    Value diff = ir.sub(lhs, rhs);
    Value zero = ir.int_immediate_number(ir.i64_type(), 0);
    Value neg = ir.lt(diff, zero);
    Value clamped = ir.select(neg, zero, diff);
    store_values_at(ir, st, current_now, clamped);
    Value next_cur = ir.add(current_now, ir.int_immediate_number(ir.i32_type(), 1));
    ir.store_variable(st.current_var, next_cur);
    ir.make_inst(spv::OpBranch, exec_merge);
  }
  // Mul
  emit_binary_advance(case_mul, [&](Value a, Value b) { return ir.mul(a, b); });
  // Max
  ir.start_label(case_max);
  {
    Value lhs = load_values_at(ir, st, op_a_i32);
    Value rhs = load_values_at(ir, st, op_b_i32);
    Value a_gt = ir.gt(lhs, rhs);
    Value v = ir.select(a_gt, lhs, rhs);
    store_values_at(ir, st, current_now, v);
    Value next_cur = ir.add(current_now, ir.int_immediate_number(ir.i32_type(), 1));
    ir.store_variable(st.current_var, next_cur);
    ir.make_inst(spv::OpBranch, exec_merge);
  }

  // BoundVariable
  ir.start_label(case_bv);
  {
    Value v = load_scope_at(ir, st, var_id_i32);
    store_values_at(ir, st, current_now, v);
    Value next_cur = ir.add(current_now, ir.int_immediate_number(ir.i32_type(), 1));
    ir.store_variable(st.current_var, next_cur);
    ir.make_inst(spv::OpBranch, exec_merge);
  }

  // ExternalTensorRead
  ir.start_label(case_etr);
  {
    Value prim_dt_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffPrimDt);
    Value prim_dt_u32 = load_buf_u32(ir, st.bytecode_buf, prim_dt_idx);
    Value prim_dt_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), prim_dt_u32);

    Value arg_off_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffArgBufferOffset);
    Value arg_off_u32 = load_buf_u32(ir, st.bytecode_buf, arg_off_idx);
    Value arg_off_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), arg_off_u32);
    // arg_buffer word index = arg_off / 4 (since arg buffer viewed as u32[])
    Value arg_word_i32 = ir.make_value(spv::OpShiftRightArithmetic, ir.i32_type(), arg_off_i32,
                                       ir.int_immediate_number(ir.i32_type(), 2));
    Value arg_word_u32 = ir.cast(ir.u32_type(), arg_word_i32);
    Value data_ptr_i64 = load_buf_i64(ir, st.args_buf, arg_word_u32);
    Value data_ptr_u64 = ir.make_value(spv::OpBitcast, ir.u64_type(), data_ptr_i64);

    Value indices_offset_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffIndicesOffset);
    Value indices_offset_u32 = load_buf_u32(ir, st.bytecode_buf, indices_offset_idx);
    Value indices_offset_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), indices_offset_u32);
    Value indices_count_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffIndicesCount);
    Value indices_count_u32 = load_buf_u32(ir, st.bytecode_buf, indices_count_idx);
    Value indices_count_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), indices_count_u32);

    Value indices_base_word = ir.load_variable(st.indices_base_word_var, ir.u32_type());
    // Indices table matches `kFieldLoad`'s `[idx_a_raw, elem_stride_a]` pair layout (pre-computed on host
    // by `encode_subtree` from the launch context's ndarray shape), so share `compute_field_load_elem_index`
    // rather than the stride-1 helper: without the per-axis multiply a multi-dim `a[i, j]` read would pick
    // up `a_flat[i + j]` instead of `a_flat[i * shape[1] + j]` and the sizer's inner loop max collapses to
    // a spurious low value, tripping an `Adstack overflow` at the next `qd.sync()`.
    Value linear_i32 = compute_field_load_elem_index(ir, st, indices_base_word, indices_offset_i32, indices_count_i32);
    Value elem_i64 = emit_psb_load_i64(ir, data_ptr_u64, linear_i32, prim_dt_i32);
    store_values_at(ir, st, current_now, elem_i64);
    Value next_cur = ir.add(current_now, ir.int_immediate_number(ir.i32_type(), 1));
    ir.store_variable(st.current_var, next_cur);
    ir.make_inst(spv::OpBranch, exec_merge);
  }

  // FieldLoad: same PSB load mechanism as ExternalTensorRead, and the indices table uses the same
  // `[idx_0_raw, elem_stride_0, ...]` pair layout - both branches share `compute_field_load_elem_index`.
  // The only distinction is where the base pointer comes from: for FieldLoad it is pre-computed on host as
  // `snode_root_buffer_psb + place_byte_offset_in_root` and stashed in the node's `const_value` slot, for
  // ETR it is read out of the kernel arg buffer at `arg_buffer_offset`.
  ir.start_label(case_fl);
  {
    Value prim_dt_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffPrimDt);
    Value prim_dt_u32 = load_buf_u32(ir, st.bytecode_buf, prim_dt_idx);
    Value prim_dt_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), prim_dt_u32);

    Value const_lo_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffConstLo);
    Value base_i64 = load_buf_i64(ir, st.bytecode_buf, const_lo_idx);
    Value base_u64 = ir.make_value(spv::OpBitcast, ir.u64_type(), base_i64);

    Value indices_offset_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffIndicesOffset);
    Value indices_offset_u32 = load_buf_u32(ir, st.bytecode_buf, indices_offset_idx);
    Value indices_offset_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), indices_offset_u32);
    Value indices_count_idx = node_field_word_idx(ir, nodes_base_word, current_now, kNodeOffIndicesCount);
    Value indices_count_u32 = load_buf_u32(ir, st.bytecode_buf, indices_count_idx);
    Value indices_count_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), indices_count_u32);

    Value indices_base_word = ir.load_variable(st.indices_base_word_var, ir.u32_type());
    Value elem_idx_i32 =
        compute_field_load_elem_index(ir, st, indices_base_word, indices_offset_i32, indices_count_i32);
    Value elem_i64 = emit_psb_load_i64(ir, base_u64, elem_idx_i32, prim_dt_i32);
    store_values_at(ir, st, current_now, elem_i64);
    Value next_cur = ir.add(current_now, ir.int_immediate_number(ir.i32_type(), 1));
    ir.store_variable(st.current_var, next_cur);
    ir.make_inst(spv::OpBranch, exec_merge);
  }

  // MaxOverRange
  ir.start_label(case_mor);
  {
    Value begin_i64 = load_values_at(ir, st, op_a_i32);
    Value end_i64 = load_values_at(ir, st, op_b_i32);
    Value empty = ir.ge(begin_i64, end_i64);

    Label empty_lbl = ir.new_label();
    Label nonempty_lbl = ir.new_label();
    Label mor_merge = ir.new_label();
    ir.make_inst(spv::OpSelectionMerge, mor_merge, spv::SelectionControlMaskNone);
    ir.make_inst(spv::OpBranchConditional, empty, empty_lbl, nonempty_lbl);

    ir.start_label(empty_lbl);
    {
      Value zero_i64 = ir.int_immediate_number(ir.i64_type(), 0);
      store_values_at(ir, st, current_now, zero_i64);
      Value next_cur = ir.add(current_now, ir.int_immediate_number(ir.i32_type(), 1));
      ir.store_variable(st.current_var, next_cur);
      ir.make_inst(spv::OpBranch, mor_merge);
    }

    ir.start_label(nonempty_lbl);
    {
      // scope[var_id] = begin
      store_scope_at(ir, st, var_id_i32, begin_i64);
      // Push pending frame: pending[sp] = {...}; sp += 1. `pending_end_arr` is clamped to `begin` when the iteration
      // count exceeds the cap, so the advance loop walks zero iterations and the dispatch returns within bounded time
      // even on the worst-case shape; the cap-hit also writes 1 into the trailing overflow-flag slot of `metadata_buf`,
      // and the host post-readback raises a `QuadrantsAssertionError` when the slot is non-zero. Matches the host
      // evaluator's `QD_ERROR_IF` in `adstack_size_expr_eval.cpp::evaluate_node` and the LLVM device sizer's
      // `scope.overflow_observed` path. Recognized `MaxOverRange` shapes are dispatched in parallel by the max-reducer
      // and substituted to a `Const` before the sizer walks the tree, so this path is reachable only for out-of-grammar
      // shapes whose iteration count exceeds the cap.
      constexpr int64_t kMaxOverRangeIterations = int64_t{1} << 24;
      Value cap_delta = ir.int_immediate_number(ir.i64_type(), kMaxOverRangeIterations);
      Value cap_end = ir.add(begin_i64, cap_delta);
      Value end_gt_cap = ir.gt(end_i64, cap_end);
      // Cap-hit collapses the walk: `effective_end = begin` so no iterations run. The overflow flag below is the signal
      // the host actually consumes; the cached `max_size` value falls through to its `max(_, 1)` floor and the heap is
      // never used because the host raises before the main kernel launches.
      Value effective_end = ir.select(end_gt_cap, begin_i64, end_i64);

      // Cap-hit overflow signal. Single-threaded dispatch, so a plain store rather than an atomic suffices. The slot is
      // initialised to 0 by the host before dispatch; the value sticks at 1 for the remainder of the dispatch once any
      // `MaxOverRange` walk in this task lands here, and the host post-readback path picks it up.
      Label cap_then = ir.new_label();
      Label cap_skip = ir.new_label();
      Label cap_merge = ir.new_label();
      ir.make_inst(spv::OpSelectionMerge, cap_merge, spv::SelectionControlMaskNone);
      ir.make_inst(spv::OpBranchConditional, end_gt_cap, cap_then, cap_skip);
      ir.start_label(cap_then);
      {
        Value overflow_word = ir.load_variable(st.overflow_flag_word_var, ir.u32_type());
        store_buf_u32(ir, st.metadata_buf, overflow_word, ir.uint_immediate_number(ir.u32_type(), 1u));
        ir.make_inst(spv::OpBranch, cap_merge);
      }
      ir.start_label(cap_skip);
      ir.make_inst(spv::OpBranch, cap_merge);
      ir.start_label(cap_merge);
      Value sp_val = ir.load_variable(st.sp_var, ir.i32_type());
      ir.store_variable(array_i32_access_ptr(ir, st.scratch_i32_buf, kI32BasePendingMorIdx, sp_val), current_now);
      Value body_start = ir.add(op_b_i32, ir.int_immediate_number(ir.i32_type(), 1));
      ir.store_variable(array_i32_access_ptr(ir, st.scratch_i32_buf, kI32BasePendingBodyStart, sp_val), body_start);
      ir.store_variable(array_i32_access_ptr(ir, st.scratch_i32_buf, kI32BasePendingBodyEnd, sp_val), body_node_i32);
      ir.store_variable(array_i64_access_ptr(ir, st.scratch_i64_buf, kI64BasePendingCurI, sp_val), begin_i64);
      ir.store_variable(array_i64_access_ptr(ir, st.scratch_i64_buf, kI64BasePendingEnd, sp_val), effective_end);
      ir.store_variable(array_i32_access_ptr(ir, st.scratch_i32_buf, kI32BasePendingVarId, sp_val), var_id_i32);
      ir.store_variable(array_i64_access_ptr(ir, st.scratch_i64_buf, kI64BasePendingMaxAccum, sp_val),
                        ir.int_immediate_number(ir.i64_type(), 0));
      ir.store_variable(array_i32_access_ptr(ir, st.scratch_i32_buf, kI32BasePendingSavedMaxK, sp_val), max_k_now);

      Value new_sp = ir.add(sp_val, ir.int_immediate_number(ir.i32_type(), 1));
      ir.store_variable(st.sp_var, new_sp);

      // current = body_start; max_k = body_node
      ir.store_variable(st.current_var, body_start);
      ir.store_variable(st.max_k_var, body_node_i32);

      ir.make_inst(spv::OpBranch, mor_merge);
    }

    ir.start_label(mor_merge);
    ir.make_inst(spv::OpBranch, exec_merge);
  }

  // default: skip the node (advance current) - should not happen with a well-formed bytecode.
  ir.start_label(case_default);
  {
    Value next_cur = ir.add(current_now, ir.int_immediate_number(ir.i32_type(), 1));
    ir.store_variable(st.current_var, next_cur);
    ir.make_inst(spv::OpBranch, exec_merge);
  }

  ir.start_label(exec_merge);
  ir.make_inst(spv::OpBranch, body_merge);

  ir.start_label(body_merge);
  ir.make_inst(spv::OpBranch, cont);

  ir.start_label(cont);
  ir.make_inst(spv::OpBranch, head);

  ir.start_label(merge);
}

}  // namespace

std::vector<uint32_t> build_adstack_sizer_spirv(Arch arch, const DeviceCapabilityConfig *caps) {
  if (!caps->get(DeviceCapability::spirv_has_physical_storage_buffer)) {
    return {};
  }
  if (!caps->get(DeviceCapability::spirv_has_int64)) {
    return {};
  }
  // `emit_psb_load_i64` materialises a per-element-type switch that unconditionally calls
  // `ir.i8_type()` / `ir.u8_type()` / `ir.i16_type()` / `ir.u16_type()`. On a Vulkan device that
  // advertises PSB + Int64 but not `shaderInt8` (VK_KHR_shader_float16_int8) or `shaderInt16` (a
  // core feature that is optional on some profiles), those accessors return a default-constructed
  // `SType` (id 0), which `spirv-val` rejects and Vulkan drivers refuse at pipeline creation.
  // Hard-error the launcher through the empty-return path so the caller surfaces the expected
  // "legacy device missing a required hardware feature" diagnostic rather than silently shipping
  // invalid SPIR-V.
  if (!caps->get(DeviceCapability::spirv_has_int8)) {
    return {};
  }
  if (!caps->get(DeviceCapability::spirv_has_int16)) {
    return {};
  }

  IRBuilder ir(arch, caps);
  // `init_header` already calls `init_pre_defs` at the end; invoking both would duplicate every primitive
  // type declaration and trip `spirv-val`'s "Duplicate non-aggregate type declarations" check.
  ir.init_header();

  // Storage-buffer bindings (set 0): bytecode in, metadata out, args in. Using `buffer_argument` because
  // all three are plain uint32[] arrays. The two scratch buffers (binding 3 = i64 scratch, binding 4 = i32
  // scratch) host the per-invocation interpreter state. Hosting that state in `StorageBuffer` SSBOs rather
  // than `Function`-storage `OpVariable`s sidesteps Blackwell-class NVIDIA driver per-thread private memory
  // caps that fail `vkCreateComputePipelines` with `VK_ERROR_UNKNOWN` once the cumulative stack-frame
  // exceeds ~32 KiB. The sizer dispatches `1x1x1` so cross-thread aliasing through the scratch SSBOs is
  // impossible; the host launcher allocates the scratch buffers fresh per `GfxRuntime` and binds them on
  // every dispatch.
  Value bytecode_buf = ir.buffer_argument(ir.u32_type(), 0, 0, "adstack_sizer_bytecode");
  Value metadata_buf = ir.buffer_argument(ir.u32_type(), 0, 1, "adstack_sizer_metadata");
  Value args_buf = ir.buffer_argument(ir.u32_type(), 0, 2, "adstack_sizer_args");
  Value scratch_i64_buf = ir.buffer_argument(ir.i64_type(), 0, 3, "adstack_sizer_scratch_i64");
  Value scratch_i32_buf = ir.buffer_argument(ir.i32_type(), 0, 4, "adstack_sizer_scratch_i32");

  Value main_func = ir.new_function();
  ir.start_function(main_func);
  ir.set_work_group_size({1, 1, 1});

  ShaderState st;
  st.bytecode_buf = bytecode_buf;
  st.metadata_buf = metadata_buf;
  st.args_buf = args_buf;
  st.scratch_i64_buf = scratch_i64_buf;
  st.scratch_i32_buf = scratch_i32_buf;

  st.current_var = ir.alloca_variable(ir.i32_type());
  st.max_k_var = ir.alloca_variable(ir.i32_type());
  st.sp_var = ir.alloca_variable(ir.i32_type());
  st.nodes_base_word_var = ir.alloca_variable(ir.u32_type());
  st.indices_base_word_var = ir.alloca_variable(ir.u32_type());

  // Zero-initialise the bound-variable scope. SPIR-V function-storage allocas are *not* required to be
  // zero-initialised (they are effectively `OpUndef`), and the outer linear pre-order walk crosses through
  // the body subtree of every `MaxOverRange` once *before* the MOR frame binds the variable - so every
  // `BoundVariable` / `ExternalTensorRead` on that first pass reads `scope[var_id]`. With undefined scope
  // those reads can produce massive indices that point outside the ndarray, which on Metal surfaces as a
  // hung dispatch (the GPU page-faults on PSB load and the command buffer never completes). The iteration
  // under the real MOR frame overwrites `values[body_idx]` with the correct answer, so a zero-valued
  // spurious eval of the body is wasted work rather than a correctness hazard; we only need the read to
  // land on a valid ndarray element. Zero is always valid for any non-empty shape, which the compile-time
  // adstack grammar already requires (empty-shape ranges collapse to `kConst 0` during pre-pass).
  Value zero_i64_for_scope_init = ir.int_immediate_number(ir.i64_type(), 0);
  for (int vi = 0; vi < kMaxVars; ++vi) {
    Value idx = ir.int_immediate_number(ir.i32_type(), vi);
    ir.store_variable(array_i64_access_ptr(ir, st.scratch_i64_buf, kI64BaseScopeArr, idx), zero_i64_for_scope_init);
  }

  // Read header: n_stacks, total_nodes.
  Value n_stacks_u32 = load_buf_u32(ir, bytecode_buf, ir.uint_immediate_number(ir.u32_type(), kHeaderOffNStacks));
  Value total_nodes_u32 = load_buf_u32(ir, bytecode_buf, ir.uint_immediate_number(ir.u32_type(), kHeaderOffTotalNodes));

  // Cache the trailing overflow-flag slot's word index. The metadata layout is `[stride_float, stride_int, off0, max0,
  // off1, max1, ..., overflow_flag]` so the slot lives at index `2 + 2 * n_stacks`. The walker writes 1 here on a
  // cap-hit (see `kMaxOverRangeIterations` branch in the per-stack tree-eval loop); the host post-readback in
  // `adstack_sizer_launch.cpp` checks the slot and raises if non-zero.
  st.overflow_flag_word_var = ir.alloca_variable(ir.u32_type());
  Value overflow_word_idx = ir.add(ir.uint_immediate_number(ir.u32_type(), 2u),
                                   ir.mul(n_stacks_u32, ir.uint_immediate_number(ir.u32_type(), 2u)));
  ir.store_variable(st.overflow_flag_word_var, overflow_word_idx);

  // Word-offsets inside the bytecode buffer for the nodes and indices arrays.
  Value header_words_u32 = ir.uint_immediate_number(ir.u32_type(), kHeaderWords);
  Value stack_header_words_u32 = ir.uint_immediate_number(ir.u32_type(), kStackHeaderWords);
  Value node_words_u32 = ir.uint_immediate_number(ir.u32_type(), kNodeWords);

  Value nodes_base = ir.add(header_words_u32, ir.mul(n_stacks_u32, stack_header_words_u32));
  Value indices_base = ir.add(nodes_base, ir.mul(total_nodes_u32, node_words_u32));
  ir.store_variable(st.nodes_base_word_var, nodes_base);
  ir.store_variable(st.indices_base_word_var, indices_base);

  // Per-stack loop variables.
  Value stack_i_var = ir.alloca_variable(ir.u32_type());
  ir.store_variable(stack_i_var, ir.uint_immediate_number(ir.u32_type(), 0));
  Value running_off_f_var = ir.alloca_variable(ir.u32_type());
  ir.store_variable(running_off_f_var, ir.uint_immediate_number(ir.u32_type(), 0));
  Value running_off_i_var = ir.alloca_variable(ir.u32_type());
  ir.store_variable(running_off_i_var, ir.uint_immediate_number(ir.u32_type(), 0));
  st.tree_start_var = ir.alloca_variable(ir.i32_type());
  ir.store_variable(st.tree_start_var, ir.int_immediate_number(ir.i32_type(), 0));

  // Per-stack main loop.
  Label stacks_head = ir.new_label();
  Label stacks_body = ir.new_label();
  Label stacks_cont = ir.new_label();
  Label stacks_merge = ir.new_label();
  ir.make_inst(spv::OpBranch, stacks_head);

  ir.start_label(stacks_head);
  Value stack_i_now = ir.load_variable(stack_i_var, ir.u32_type());
  Value more_stacks = ir.lt(stack_i_now, n_stacks_u32);
  ir.make_inst(spv::OpLoopMerge, stacks_merge, stacks_cont, spv::LoopControlMaskNone);
  ir.make_inst(spv::OpBranchConditional, more_stacks, stacks_body, stacks_merge);

  ir.start_label(stacks_body);
  {
    // Read stack header fields.
    Value sh_base_word = ir.add(header_words_u32, ir.mul(stack_i_now, stack_header_words_u32));
    Value root_idx_raw = load_buf_u32(
        ir, bytecode_buf, ir.add(sh_base_word, ir.uint_immediate_number(ir.u32_type(), kStackOffRootNodeIdx)));
    Value root_idx_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), root_idx_raw);
    Value max_size_ct = load_buf_u32(
        ir, bytecode_buf, ir.add(sh_base_word, ir.uint_immediate_number(ir.u32_type(), kStackOffMaxSizeCompileTime)));
    Value heap_kind = load_buf_u32(ir, bytecode_buf,
                                   ir.add(sh_base_word, ir.uint_immediate_number(ir.u32_type(), kStackOffHeapKind)));

    // Resolve `max_size`.
    Value has_tree = ir.ge(root_idx_i32, ir.int_immediate_number(ir.i32_type(), 0));
    Label tree_lbl = ir.new_label();
    Label no_tree_lbl = ir.new_label();
    Label resolve_merge = ir.new_label();
    ir.make_inst(spv::OpSelectionMerge, resolve_merge, spv::SelectionControlMaskNone);
    ir.make_inst(spv::OpBranchConditional, has_tree, tree_lbl, no_tree_lbl);

    ir.start_label(no_tree_lbl);
    // max_size = max(max_size_compile_time, 1)
    Value one_u32 = ir.uint_immediate_number(ir.u32_type(), 1);
    Value is_zero = ir.eq(max_size_ct, ir.uint_immediate_number(ir.u32_type(), 0));
    Value no_tree_max = ir.select(is_zero, one_u32, max_size_ct);
    Label no_tree_end = ir.current_label();
    ir.make_inst(spv::OpBranch, resolve_merge);

    ir.start_label(tree_lbl);
    // Initialise eval state and run the tree loop.
    Value tree_start_val = ir.load_variable(st.tree_start_var, ir.i32_type());
    ir.store_variable(st.current_var, tree_start_val);
    ir.store_variable(st.max_k_var, root_idx_i32);
    ir.store_variable(st.sp_var, ir.int_immediate_number(ir.i32_type(), 0));
    emit_tree_eval_loop(ir, st);
    // Pick up the root value and apply the lower `>= 1` guard. No upper clamp: `max_size_compile_time` is
    // the *fallback* used when there is no symbolic tree (see the `no_tree_lbl` branch), not a hard ceiling
    // on the runtime-evaluated size. An upper clamp here would silently undercount any tree whose evaluated
    // bound exceeds the structural default - exactly the scenario the per-launch SizeExpr machinery exists
    // to service - and the downstream heap path is already sized from this value, so there is no memory
    // safety reason to cap it either. Matches `llvm_runtime_executor.cpp`'s CPU branch (no upper clamp).
    Value root_val_i64 = load_values_at(ir, st, root_idx_i32);
    Value one_i64 = ir.int_immediate_number(ir.i64_type(), 1);
    Value too_small = ir.lt(root_val_i64, one_i64);
    Value clamped = ir.select(too_small, one_i64, root_val_i64);
    // Cast to u32 for metadata output.
    Value tree_max_u32 = ir.cast(ir.u32_type(), clamped);
    Label tree_end = ir.current_label();
    ir.make_inst(spv::OpBranch, resolve_merge);

    ir.start_label(resolve_merge);
    PhiValue max_size_phi = ir.make_phi(ir.u32_type(), 2);
    max_size_phi.set_incoming(0, no_tree_max, no_tree_end);
    max_size_phi.set_incoming(1, tree_max_u32, tree_end);
    Value max_size_u32 = Value(max_size_phi);

    // Route to float/int running-offset and write metadata entries.
    Value slot_base_word = ir.add(ir.uint_immediate_number(ir.u32_type(), 2u),
                                  ir.mul(stack_i_now, ir.uint_immediate_number(ir.u32_type(), 2u)));
    Value slot_off_word = slot_base_word;
    Value slot_max_word = ir.add(slot_base_word, ir.uint_immediate_number(ir.u32_type(), 1u));

    Value heap_is_float = ir.eq(heap_kind, ir.uint_immediate_number(ir.u32_type(), 0));
    Label float_lbl = ir.new_label();
    Label int_lbl = ir.new_label();
    Label heap_merge = ir.new_label();
    ir.make_inst(spv::OpSelectionMerge, heap_merge, spv::SelectionControlMaskNone);
    ir.make_inst(spv::OpBranchConditional, heap_is_float, float_lbl, int_lbl);

    ir.start_label(float_lbl);
    {
      Value off_now = ir.load_variable(running_off_f_var, ir.u32_type());
      store_buf_u32(ir, metadata_buf, slot_off_word, off_now);
      store_buf_u32(ir, metadata_buf, slot_max_word, max_size_u32);
      // Float heap: primal + adjoint interleaved, so advance by 2 * max_size.
      Value add = ir.mul(ir.uint_immediate_number(ir.u32_type(), 2u), max_size_u32);
      ir.store_variable(running_off_f_var, ir.add(off_now, add));
      ir.make_inst(spv::OpBranch, heap_merge);
    }

    ir.start_label(int_lbl);
    {
      Value off_now = ir.load_variable(running_off_i_var, ir.u32_type());
      store_buf_u32(ir, metadata_buf, slot_off_word, off_now);
      store_buf_u32(ir, metadata_buf, slot_max_word, max_size_u32);
      ir.store_variable(running_off_i_var, ir.add(off_now, max_size_u32));
      ir.make_inst(spv::OpBranch, heap_merge);
    }

    ir.start_label(heap_merge);
    // Advance tree_start if this stack had a tree.
    Label has_tree2 = ir.new_label();
    Label no_tree2 = ir.new_label();
    Label adv_merge = ir.new_label();
    Value has_tree_again = ir.ge(root_idx_i32, ir.int_immediate_number(ir.i32_type(), 0));
    ir.make_inst(spv::OpSelectionMerge, adv_merge, spv::SelectionControlMaskNone);
    ir.make_inst(spv::OpBranchConditional, has_tree_again, has_tree2, no_tree2);

    ir.start_label(has_tree2);
    Value next_tree_start = ir.add(root_idx_i32, ir.int_immediate_number(ir.i32_type(), 1));
    ir.store_variable(st.tree_start_var, next_tree_start);
    ir.make_inst(spv::OpBranch, adv_merge);

    ir.start_label(no_tree2);
    ir.make_inst(spv::OpBranch, adv_merge);

    ir.start_label(adv_merge);
    ir.make_inst(spv::OpBranch, stacks_cont);
  }

  ir.start_label(stacks_cont);
  Value stack_i_loaded = ir.load_variable(stack_i_var, ir.u32_type());
  Value stack_i_next = ir.add(stack_i_loaded, ir.uint_immediate_number(ir.u32_type(), 1));
  ir.store_variable(stack_i_var, stack_i_next);
  ir.make_inst(spv::OpBranch, stacks_head);

  ir.start_label(stacks_merge);

  // Finalise: write the running strides into metadata[0] / metadata[1].
  Value final_off_f = ir.load_variable(running_off_f_var, ir.u32_type());
  Value final_off_i = ir.load_variable(running_off_i_var, ir.u32_type());
  store_buf_u32(ir, metadata_buf, ir.uint_immediate_number(ir.u32_type(), 0), final_off_f);
  store_buf_u32(ir, metadata_buf, ir.uint_immediate_number(ir.u32_type(), 1), final_off_i);

  ir.make_inst(spv::OpReturn);
  ir.make_inst(spv::OpFunctionEnd);

  std::vector<Value> entry_args = {bytecode_buf, metadata_buf, args_buf, scratch_i64_buf, scratch_i32_buf};
  // Entry point name MUST be "main" to match the rest of the Quadrants SPIR-V codegen
  // (see `spirv_codegen.cpp:commit_kernel_function(..., "main", ...)`). The Metal RHI renames
  // the SPIR-V `main` to `main0` during MSL cross-compilation and then looks up the compute
  // function by that exact name (`metal_device.mm: get_mtl_function(..., "main0")`). Any
  // other entry point name produces `nil` from `newFunctionWithName:` and trips the hard
  // `computeFunction must not be nil` assertion inside Metal at pipeline-creation time.
  ir.commit_kernel_function(main_func, "main", entry_args, {1, 1, 1});

  return ir.finalize();
}

}  // namespace quadrants::lang::spirv
