#include "quadrants/codegen/spirv/adstack_max_reducer_shader.h"

#include "quadrants/codegen/spirv/spirv_ir_builder.h"
#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/ir/type.h"

namespace quadrants::lang::spirv {

namespace {

// Number of u32 words per `AdStackSizeExprDeviceNode` in the bytecode buffer. The host encoder writes the POD
// directly through a `memcpy`-style copy; the shader reads field-by-field at compile-time-known word offsets within
// the per-node 12-word slot. Kept as a single named constant so a future change to the device-node layout (e.g.
// dropping `_pad0`) only needs to touch one place. Mirrors the sizer's per-node walk in
// `adstack_sizer_shader.cpp` which uses the same convention.
constexpr uint32_t kNodeWords = sizeof(AdStackSizeExprDeviceNode) / 4u;
static_assert(sizeof(AdStackSizeExprDeviceNode) % 4u == 0u,
              "AdStackSizeExprDeviceNode must be a multiple of 4 bytes for direct u32[] indexing");

// Field offsets within an `AdStackSizeExprDeviceNode` slot, in u32 words. Keep in sync with the struct definition in
// `quadrants/ir/adstack_size_expr_device.h`. Read fields by `slot_base_word + kField...` to keep the shader IR
// straight-line and easy to read against the host POD.
constexpr uint32_t kNodeWordKind = 0;
constexpr uint32_t kNodeWordOperandA = 1;
constexpr uint32_t kNodeWordOperandB = 2;
// `kNodeWordVarId = 4` is reserved by the device-node POD for `kBoundVariable` resolution; the max-reducer
// substitutes its single bound variable directly without consulting `var_id` so the slot is unused here. Listed in
// this comment instead of as a `constexpr` to keep the symbol clean (`-Werror=unused-const-variable`).
constexpr uint32_t kNodeWordPrimDt = 5;
constexpr uint32_t kNodeWordArgBufferOffset = 6;
constexpr uint32_t kNodeWordIndicesOffset = 7;
constexpr uint32_t kNodeWordIndicesCount = 8;
// `const_value` is a 64-bit field at words 10/11; we always read both halves through `load_buf_i64` which adds 1 to
// the lo word index internally, so we only name the lo offset here.
constexpr uint32_t kNodeWordConstValueLo = 10;

// Small helper: read one uint32 word from a storage-buffer-backed uint32[] at the given scalar index. Mirrors the
// same-named helper in `adstack_bound_reducer_shader.cpp` and `adstack_sizer_shader.cpp`; kept local to this TU so the
// shader's symbol set stays self-contained and the helper inlines without cross-file linkage.
Value load_buf_u32(IRBuilder &ir, Value buffer, Value word_idx) {
  Value ptr = ir.struct_array_access(ir.u32_type(), buffer, word_idx);
  return ir.load_variable(ptr, ir.u32_type());
}

// Load one i32 word from a storage-buffer-backed uint32[] at scalar index `word_idx`, bitcast-reinterpreted as i32.
// The host encoder writes the PODs verbatim, so signed fields like `operand_a` round-trip through the u32 SSBO via
// little-endian bitcast. Returns an i32 SSA value.
Value load_buf_i32(IRBuilder &ir, Value buffer, Value word_idx) {
  Value u = load_buf_u32(ir, buffer, word_idx);
  return ir.make_value(spv::OpBitcast, ir.i32_type(), u);
}

// Load one i64 from two adjacent u32 words at `lo_word_idx` and `lo_word_idx + 1`. Used for `kConst` nodes whose
// `const_value` field straddles two u32 slots in the bytecode buffer (12-word node POD layout). Returned as i64.
Value load_buf_i64(IRBuilder &ir, Value buffer, Value lo_word_idx) {
  Value lo = load_buf_u32(ir, buffer, lo_word_idx);
  Value hi = load_buf_u32(ir, buffer, ir.add(lo_word_idx, ir.uint_immediate_number(ir.u32_type(), 1u)));
  Value lo64 = ir.cast(ir.u64_type(), lo);
  Value hi64 = ir.cast(ir.u64_type(), hi);
  Value shift = ir.uint_immediate_number(ir.u64_type(), 32u);
  Value hi_shifted = ir.make_value(spv::OpShiftLeftLogical, ir.u64_type(), hi64, shift);
  Value combined_u64 = ir.make_value(spv::OpBitwiseOr, ir.u64_type(), lo64, hi_shifted);
  return ir.make_value(spv::OpBitcast, ir.i64_type(), combined_u64);
}

// Assemble a u64 from two adjacent little-endian u32 words at `base_word_idx` and `base_word_idx + 1`. Used to
// reconstruct ndarray data pointers from the kernel arg buffer for `kExternalTensorRead`.
Value load_arg_buf_u64_ptr(IRBuilder &ir, Value buffer, Value base_word_idx) {
  Value lo = load_buf_u32(ir, buffer, base_word_idx);
  Value hi = load_buf_u32(ir, buffer, ir.add(base_word_idx, ir.uint_immediate_number(ir.u32_type(), 1u)));
  Value lo64 = ir.cast(ir.u64_type(), lo);
  Value hi64 = ir.cast(ir.u64_type(), hi);
  Value shift = ir.uint_immediate_number(ir.u64_type(), 32u);
  Value hi_shifted = ir.make_value(spv::OpShiftLeftLogical, ir.u64_type(), hi64, shift);
  return ir.make_value(spv::OpBitwiseOr, ir.u64_type(), lo64, hi_shifted);
}

// Physical-Storage-Buffer load of one scalar of `load_ty` width `elem_size` bytes at byte offset `byte_off_u64` from
// `base_u64`. Mirrors the wrapper-struct PSB load pattern in `adstack_sizer_shader.cpp::psb_load_scalar` and
// `adstack_bound_reducer_shader.cpp::psb_load_u32_at_byte_off`.
Value psb_load_scalar_at_byte_off(IRBuilder &ir,
                                  Value base_u64,
                                  Value byte_off_u64,
                                  const SType &load_ty,
                                  uint32_t elem_alignment) {
  Value target_u64 = ir.add(base_u64, byte_off_u64);
  SType ptr_elem_type = ir.get_pointer_type(load_ty, spv::StorageClassPhysicalStorageBuffer);
  std::vector<std::tuple<SType, std::string, size_t>> members = {{load_ty, "_m0", 0}};
  SType wrapper_struct = ir.create_struct_type(members);
  SType ptr_struct_type = ir.get_pointer_type(wrapper_struct, spv::StorageClassPhysicalStorageBuffer);
  Value struct_ptr = ir.make_value(spv::OpConvertUToPtr, ptr_struct_type, target_u64);
  Value scalar_ptr = ir.make_value(spv::OpAccessChain, ptr_elem_type, struct_ptr, ir.const_i32_zero_);
  Value scalar = ir.new_value(load_ty, ValueKind::kNormal);
  ir.make_inst(spv::OpLoad, load_ty, scalar, scalar_ptr, spv::MemoryAccessAlignedMask, elem_alignment);
  return scalar;
}

// Switch on `prim_dt` (a `PrimitiveTypeID` value) and emit the matching PSB load + sign/zero-extend to i64. The
// recognizer only emits integer leaves (`recognize_adstack_max_reducer_specs` rejects float-typed bodies), so the float
// arms are unreachable and not emitted. Element index is in elements (not bytes); the per-dtype load multiplies by the
// element size internally.
Value emit_psb_load_i64_int_only(IRBuilder &ir, Value data_ptr_u64, Value linear_i32, Value prim_dt_i32) {
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

  Value linear_u64 = ir.cast(ir.u64_type(), ir.make_value(spv::OpBitcast, ir.u32_type(), linear_i32));
  Value result_var = ir.alloca_variable(ir.i64_type());
  ir.store_variable(result_var, ir.int_immediate_number(ir.i64_type(), 0));

  ir.make_inst(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpSwitch, prim_dt_i32, case_default,               //
               static_cast<uint32_t>(PrimitiveTypeID::i8), case_i8,    //
               static_cast<uint32_t>(PrimitiveTypeID::i16), case_i16,  //
               static_cast<uint32_t>(PrimitiveTypeID::i32), case_i32,  //
               static_cast<uint32_t>(PrimitiveTypeID::i64), case_i64,  //
               static_cast<uint32_t>(PrimitiveTypeID::u8), case_u8,    //
               static_cast<uint32_t>(PrimitiveTypeID::u16), case_u16,  //
               static_cast<uint32_t>(PrimitiveTypeID::u32), case_u32,  //
               static_cast<uint32_t>(PrimitiveTypeID::u64), case_u64);

  auto emit_int_case = [&](Label lbl, const SType &load_ty, uint32_t elem_size, bool is_signed) {
    ir.start_label(lbl);
    Value byte_off = ir.mul(linear_u64, ir.uint_immediate_number(ir.u64_type(), elem_size));
    Value v = psb_load_scalar_at_byte_off(ir, data_ptr_u64, byte_off, load_ty, elem_size);
    Value v_i64;
    if (load_ty.id == ir.i64_type().id) {
      v_i64 = v;
    } else if (load_ty.id == ir.u64_type().id) {
      v_i64 = ir.make_value(spv::OpBitcast, ir.i64_type(), v);
    } else if (is_signed) {
      v_i64 = ir.make_value(spv::OpSConvert, ir.i64_type(), v);
    } else {
      Value v_u64 = ir.make_value(spv::OpUConvert, ir.u64_type(), v);
      v_i64 = ir.make_value(spv::OpBitcast, ir.i64_type(), v_u64);
    }
    ir.store_variable(result_var, v_i64);
    ir.make_inst(spv::OpBranch, merge);
  };

  emit_int_case(case_i8, ir.i8_type(), 1u, true);
  emit_int_case(case_i16, ir.i16_type(), 2u, true);
  emit_int_case(case_i32, ir.i32_type(), 4u, true);
  emit_int_case(case_i64, ir.i64_type(), 8u, true);
  emit_int_case(case_u8, ir.u8_type(), 1u, false);
  emit_int_case(case_u16, ir.u16_type(), 2u, false);
  emit_int_case(case_u32, ir.u32_type(), 4u, false);
  emit_int_case(case_u64, ir.u64_type(), 8u, false);

  ir.start_label(case_default);
  ir.make_inst(spv::OpBranch, merge);

  ir.start_label(merge);
  return ir.load_variable(result_var, ir.i64_type());
}

// Dynamic-index access into a Function-scope array (OpVariable with array type, allocated via `alloca_variable`).
// `IRBuilder::struct_array_access` is a buffer-only helper (asserts `kStructArrayPtr`); for Function-scope
// arrays we emit `OpAccessChain` directly with the per-element pointer type. Returns a pointer-to-element that
// can be passed to `load_variable` / `store_variable`.
Value alloca_array_access(IRBuilder &ir, Value arr_var, const SType &elem_type, Value index_i32) {
  SType elem_ptr_type = ir.get_pointer_type(elem_type, spv::StorageClassFunction);
  Value elem_ptr = ir.new_value(elem_ptr_type, ValueKind::kVariablePtr);
  ir.make_inst(spv::OpAccessChain, elem_ptr_type, elem_ptr, arr_var, index_i32);
  return elem_ptr;
}

// Compute the element index for a `kExternalTensorRead` node body leaf. The recognizer grammar restricts the indices
// table to a single axis whose raw value is `-(this_var_id + 1)` (referencing the enclosing `MaxOverRange`'s bound
// variable), but the loop is general - it walks `indices[node.indices_offset .. node.indices_offset + 2 *
// node.indices_count)` as `(idx_raw, elem_stride)` pairs and accumulates `v * stride` where `v` is the resolved
// integer index. `iter_var_i32` is the value to substitute for any `BoundVariable` reference (encoded as a negative
// `idx_raw` per the `SerializedSizeExprNode::indices` convention shared with the host evaluator). The recognizer only
// references the immediately-enclosing `MaxOverRange`'s bound var so a single substitution variable suffices; future
// grammar extensions may need a small scope array.
Value compute_external_read_elem_index(IRBuilder &ir,
                                       Value bytecode_buf,
                                       Value indices_base_word,
                                       Value indices_offset_i32,
                                       Value indices_count_i32,
                                       Value iter_var_i32) {
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
  Value k_u32 = ir.cast(ir.u32_type(), k_now);
  Value pair_base_u32 = ir.add(indices_off_u32, ir.mul(k_u32, ir.uint_immediate_number(ir.u32_type(), 2u)));
  Value idx_word_u32 = ir.add(indices_base_word, pair_base_u32);
  Value stride_word_u32 = ir.add(idx_word_u32, ir.uint_immediate_number(ir.u32_type(), 1u));
  Value idx_raw_i32 = load_buf_i32(ir, bytecode_buf, idx_word_u32);
  Value stride_i32 = load_buf_i32(ir, bytecode_buf, stride_word_u32);

  // Resolve the raw index. `idx_raw >= 0` means a constant axis index baked at encode time; `idx_raw < 0` means a
  // bound-variable reference encoded as `-(var_id + 1)` and we substitute `iter_var_i32`. The recognizer accepts only
  // one bound variable per spec (the enclosing `MaxOverRange`'s var_id) so any negative `idx_raw` resolves to the same
  // `iter_var_i32` value; we do not bother validating the encoded var_id here (the host encoder asserts it matches
  // before emitting bytecode).
  Label const_lbl = ir.new_label();
  Label var_lbl = ir.new_label();
  Label sel_merge = ir.new_label();
  Value is_const = ir.ge(idx_raw_i32, ir.int_immediate_number(ir.i32_type(), 0));
  ir.make_inst(spv::OpSelectionMerge, sel_merge, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpBranchConditional, is_const, const_lbl, var_lbl);

  ir.start_label(const_lbl);
  Value v_const_i32 = idx_raw_i32;
  Label const_end = ir.current_label();
  ir.make_inst(spv::OpBranch, sel_merge);

  ir.start_label(var_lbl);
  Value v_var_i32 = iter_var_i32;
  Label var_end = ir.current_label();
  ir.make_inst(spv::OpBranch, sel_merge);

  ir.start_label(sel_merge);
  PhiValue v = ir.make_phi(ir.i32_type(), 2);
  v.set_incoming(0, v_const_i32, const_end);
  v.set_incoming(1, v_var_i32, var_end);

  Value contribution = ir.mul(Value(v), stride_i32);
  Value acc_now = ir.load_variable(acc_var, ir.i32_type());
  ir.store_variable(acc_var, ir.add(acc_now, contribution));
  ir.make_inst(spv::OpBranch, cont);

  ir.start_label(cont);
  Value k_next = ir.add(k_now, ir.int_immediate_number(ir.i32_type(), 1));
  ir.store_variable(k_var, k_next);
  ir.make_inst(spv::OpBranch, head);

  ir.start_label(merge);
  return ir.load_variable(acc_var, ir.i32_type());
}

// Per-thread post-order interpreter for the body subtree. Iterates ascending node indices `0..body_node_count`,
// reading each node's POD from the bytecode buffer at `body_bytecode_offset_words + i * kNodeWords` and storing the
// computed i64 value into `vals[i]`. The `vals[]` array is a Function-scope OpVariable of `i64[kAdStackMaxReducerMax
// BodyNodes]`; per-thread footprint is 8 bytes/node-slot. Final result is `vals[body_node_count - 1]` (the root, since
// post-order encoding places the root last).
Value interpret_body(IRBuilder &ir,
                     Value args_buf,
                     Value bytecode_buf,
                     Value body_bytecode_offset_words,
                     Value body_indices_offset_words,
                     Value body_node_count_i32,
                     Value iter_var_i32) {
  // Function-scope per-thread value storage. Allocate as a fixed-size i64 array; the index used at store time is the
  // current node index.
  SType i64_arr_ty = ir.get_array_type(ir.i64_type(), kAdStackMaxReducerMaxBodyNodes);
  Value vals_var = ir.alloca_variable(i64_arr_ty);

  Value i_var = ir.alloca_variable(ir.i32_type());
  ir.store_variable(i_var, ir.int_immediate_number(ir.i32_type(), 0));

  Label head = ir.new_label();
  Label body_lbl = ir.new_label();
  Label cont = ir.new_label();
  Label merge = ir.new_label();

  ir.make_inst(spv::OpBranch, head);

  ir.start_label(head);
  Value i_now = ir.load_variable(i_var, ir.i32_type());
  Value loop_cond = ir.lt(i_now, body_node_count_i32);
  ir.make_inst(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
  ir.make_inst(spv::OpBranchConditional, loop_cond, body_lbl, merge);

  ir.start_label(body_lbl);
  // Compute this node's slot base word: `body_bytecode_offset_words + i * kNodeWords`.
  Value i_u32 = ir.cast(ir.u32_type(), i_now);
  Value slot_base =
      ir.add(body_bytecode_offset_words, ir.mul(i_u32, ir.uint_immediate_number(ir.u32_type(), kNodeWords)));
  Value kind_i32 =
      load_buf_i32(ir, bytecode_buf, ir.add(slot_base, ir.uint_immediate_number(ir.u32_type(), kNodeWordKind)));

  Label case_const = ir.new_label();
  Label case_bv = ir.new_label();
  Label case_etr = ir.new_label();
  Label case_add = ir.new_label();
  Label case_sub = ir.new_label();
  Label case_mul = ir.new_label();
  Label case_max = ir.new_label();
  Label case_default = ir.new_label();
  Label kind_merge = ir.new_label();

  Value computed_var = ir.alloca_variable(ir.i64_type());
  ir.store_variable(computed_var, ir.int_immediate_number(ir.i64_type(), 0));

  ir.make_inst(spv::OpSelectionMerge, kind_merge, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpSwitch, kind_i32, case_default,                                            //
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kConst), case_const,             //
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kBoundVariable), case_bv,        //
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kExternalTensorRead), case_etr,  //
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kAdd), case_add,                 //
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kSub), case_sub,                 //
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kMul), case_mul,                 //
               static_cast<uint32_t>(AdStackSizeExprDeviceKind::kMax), case_max);

  // kConst: load the i64 const_value from words [10, 11] of this slot.
  ir.start_label(case_const);
  {
    Value const_val = load_buf_i64(ir, bytecode_buf,
                                   ir.add(slot_base, ir.uint_immediate_number(ir.u32_type(), kNodeWordConstValueLo)));
    ir.store_variable(computed_var, const_val);
    ir.make_inst(spv::OpBranch, kind_merge);
  }

  // kBoundVariable: substitute `iter_var_i32` (extended to i64). The recognizer grammar binds exactly one var per body
  // (the enclosing `MaxOverRange`'s bound variable), so we do not consult `var_id` and just return `iter_var`.
  ir.start_label(case_bv);
  {
    Value iter_var_i64 = ir.make_value(spv::OpSConvert, ir.i64_type(), iter_var_i32);
    ir.store_variable(computed_var, iter_var_i64);
    ir.make_inst(spv::OpBranch, kind_merge);
  }

  // kExternalTensorRead: PSB-load the body ndarray's element at the computed linear index.
  ir.start_label(case_etr);
  {
    Value arg_word_offset = load_buf_i32(
        ir, bytecode_buf, ir.add(slot_base, ir.uint_immediate_number(ir.u32_type(), kNodeWordArgBufferOffset)));
    Value prim_dt =
        load_buf_i32(ir, bytecode_buf, ir.add(slot_base, ir.uint_immediate_number(ir.u32_type(), kNodeWordPrimDt)));
    Value indices_offset = load_buf_i32(
        ir, bytecode_buf, ir.add(slot_base, ir.uint_immediate_number(ir.u32_type(), kNodeWordIndicesOffset)));
    Value indices_count = load_buf_i32(
        ir, bytecode_buf, ir.add(slot_base, ir.uint_immediate_number(ir.u32_type(), kNodeWordIndicesCount)));
    Value linear_i32 = compute_external_read_elem_index(ir, bytecode_buf, body_indices_offset_words, indices_offset,
                                                        indices_count, iter_var_i32);
    Value arg_word_offset_u32 = ir.make_value(spv::OpBitcast, ir.u32_type(), arg_word_offset);
    Value data_ptr_u64 = load_arg_buf_u64_ptr(ir, args_buf, arg_word_offset_u32);
    Value loaded_i64 = emit_psb_load_i64_int_only(ir, data_ptr_u64, linear_i32, prim_dt);
    ir.store_variable(computed_var, loaded_i64);
    ir.make_inst(spv::OpBranch, kind_merge);
  }

  // Helper to load `vals[op_idx]` for the binary arithmetic cases. The op-index is read from the slot's `operand_a`
  // / `operand_b` fields, both of which the post-order encoding guarantees are < `i_now` so the value is already
  // computed.
  auto load_val_at = [&](uint32_t word_off) -> Value {
    Value op_idx_i32 =
        load_buf_i32(ir, bytecode_buf, ir.add(slot_base, ir.uint_immediate_number(ir.u32_type(), word_off)));
    Value ptr = alloca_array_access(ir, vals_var, ir.i64_type(), op_idx_i32);
    return ir.load_variable(ptr, ir.i64_type());
  };

  ir.start_label(case_add);
  {
    Value a = load_val_at(kNodeWordOperandA);
    Value b = load_val_at(kNodeWordOperandB);
    ir.store_variable(computed_var, ir.add(a, b));
    ir.make_inst(spv::OpBranch, kind_merge);
  }

  ir.start_label(case_sub);
  {
    Value a = load_val_at(kNodeWordOperandA);
    Value b = load_val_at(kNodeWordOperandB);
    Value diff = ir.sub(a, b);
    // Match the host evaluator's saturating-sub behaviour for `SizeExpr::Kind::Sub` (clamps to 0). Per-thread sizes
    // are non-negative by construction; signed subtraction would let a negative value poison the running max if a
    // body subtree's `arr_a[i] < arr_b[i]` for some thread.
    Value zero_i64 = ir.int_immediate_number(ir.i64_type(), 0);
    Value is_neg = ir.lt(diff, zero_i64);
    Value clamped = ir.make_value(spv::OpSelect, ir.i64_type(), is_neg, zero_i64, diff);
    ir.store_variable(computed_var, clamped);
    ir.make_inst(spv::OpBranch, kind_merge);
  }

  ir.start_label(case_mul);
  {
    Value a = load_val_at(kNodeWordOperandA);
    Value b = load_val_at(kNodeWordOperandB);
    ir.store_variable(computed_var, ir.mul(a, b));
    ir.make_inst(spv::OpBranch, kind_merge);
  }

  ir.start_label(case_max);
  {
    Value a = load_val_at(kNodeWordOperandA);
    Value b = load_val_at(kNodeWordOperandB);
    Value gt = ir.make_value(spv::OpSGreaterThan, ir.bool_type(), a, b);
    Value m = ir.make_value(spv::OpSelect, ir.i64_type(), gt, a, b);
    ir.store_variable(computed_var, m);
    ir.make_inst(spv::OpBranch, kind_merge);
  }

  ir.start_label(case_default);
  ir.make_inst(spv::OpBranch, kind_merge);

  ir.start_label(kind_merge);
  Value computed = ir.load_variable(computed_var, ir.i64_type());
  // Store into vals[i].
  Value vals_slot_ptr = alloca_array_access(ir, vals_var, ir.i64_type(), i_now);
  ir.store_variable(vals_slot_ptr, computed);
  ir.make_inst(spv::OpBranch, cont);

  ir.start_label(cont);
  Value i_next = ir.add(i_now, ir.int_immediate_number(ir.i32_type(), 1));
  ir.store_variable(i_var, i_next);
  ir.make_inst(spv::OpBranch, head);

  ir.start_label(merge);
  // Root index = body_node_count - 1.
  Value root_idx = ir.sub(body_node_count_i32, ir.int_immediate_number(ir.i32_type(), 1));
  Value root_ptr = alloca_array_access(ir, vals_var, ir.i64_type(), root_idx);
  return ir.load_variable(root_ptr, ir.i64_type());
}

}  // namespace

std::vector<uint32_t> build_adstack_max_reducer_spirv(Arch arch, const DeviceCapabilityConfig *caps) {
  if (!caps->get(DeviceCapability::spirv_has_physical_storage_buffer)) {
    return {};
  }
  if (!caps->get(DeviceCapability::spirv_has_int64)) {
    return {};
  }

  IRBuilder ir(arch, caps);
  ir.init_header();

  // Storage-buffer bindings (set 0).
  Value args_buf = ir.buffer_argument(ir.u32_type(), 0, 0, "adstack_max_reducer_args");
  Value output_buf = ir.buffer_argument(ir.u64_type(), 0, 1, "adstack_max_reducer_output");
  Value params_buf = ir.buffer_argument(ir.u32_type(), 0, 2, "adstack_max_reducer_params");
  Value bytecode_buf = ir.buffer_argument(ir.u32_type(), 0, 3, "adstack_max_reducer_bytecode");

  Value main_func = ir.new_function();
  ir.start_function(main_func);
  ir.set_work_group_size({static_cast<int>(kAdStackMaxReducerWorkgroupSize), 1, 1});

  Value gid_u32 = ir.get_global_invocation_id(0);

  // Load params at the top of `main`. spirv-opt CSEs the redundant loads if any, but the explicit hoist makes the
  // shader's data flow easier to read against the host POD.
  Value output_slot = load_buf_u32(
      ir, params_buf, ir.uint_immediate_number(ir.u32_type(), AdStackMaxReducerParams::kWordOffsetOutputSlot));
  Value length =
      load_buf_u32(ir, params_buf, ir.uint_immediate_number(ir.u32_type(), AdStackMaxReducerParams::kWordOffsetLength));
  Value begin_lo = load_buf_u32(ir, params_buf,
                                ir.uint_immediate_number(ir.u32_type(), AdStackMaxReducerParams::kWordOffsetBeginLo));
  Value begin_hi = load_buf_u32(ir, params_buf,
                                ir.uint_immediate_number(ir.u32_type(), AdStackMaxReducerParams::kWordOffsetBeginHi));
  Value body_bytecode_offset_words = load_buf_u32(
      ir, params_buf,
      ir.uint_immediate_number(ir.u32_type(), AdStackMaxReducerParams::kWordOffsetBodyBytecodeOffsetWords));
  Value body_node_count_u32 = load_buf_u32(
      ir, params_buf, ir.uint_immediate_number(ir.u32_type(), AdStackMaxReducerParams::kWordOffsetBodyNodeCount));
  Value body_indices_offset_words =
      load_buf_u32(ir, params_buf,
                   ir.uint_immediate_number(ir.u32_type(), AdStackMaxReducerParams::kWordOffsetBodyIndicesOffsetWords));

  // Trailing-workgroup bounds check. `gid >= length` threads early-return; remaining threads run the body.
  Label active_block = ir.new_label();
  Label early_return = ir.new_label();
  Label active_merge = ir.new_label();
  Value in_range = ir.lt(gid_u32, length);
  ir.make_inst(spv::OpSelectionMerge, active_merge, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpBranchConditional, in_range, active_block, early_return);

  ir.start_label(active_block);
  {
    // Reassemble `begin` as i64 and compute the per-thread bound variable: `iter_var = (i32)(gid + begin)`.
    // The recognizer grammar caps `begin + length` at i32 max (the encoder rejects specs whose closed-form bound
    // exceeds 2^31), so a 32-bit add is safe and matches the existing sizer's per-thread index width.
    Value lo64 = ir.cast(ir.u64_type(), begin_lo);
    Value hi64 = ir.cast(ir.u64_type(), begin_hi);
    Value shift = ir.uint_immediate_number(ir.u64_type(), 32u);
    Value hi_shifted = ir.make_value(spv::OpShiftLeftLogical, ir.u64_type(), hi64, shift);
    Value begin_u64 = ir.make_value(spv::OpBitwiseOr, ir.u64_type(), lo64, hi_shifted);
    Value begin_i64 = ir.make_value(spv::OpBitcast, ir.i64_type(), begin_u64);
    Value gid_i64 = ir.cast(ir.i64_type(), gid_u32);
    Value iter_var_i64 = ir.add(gid_i64, begin_i64);
    Value iter_var_i32 = ir.cast(ir.i32_type(), iter_var_i64);

    Value body_node_count_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), body_node_count_u32);
    Value result_i64 = interpret_body(ir, args_buf, bytecode_buf, body_bytecode_offset_words, body_indices_offset_words,
                                      body_node_count_i32, iter_var_i32);

    // Atomic-max into `output_buf[output_slot]`. SPIR-V requires `OpAtomicSMax` on i64; per-launch host clears the
    // slot to `INT64_MIN` (so the first matching thread wins). Memory scope = Device, semantics = Relaxed (the value
    // is consumed by the host post-`wait_idle`, no in-shader fence required).
    Value slot_ptr = ir.struct_array_access(ir.i64_type(), output_buf, output_slot);
    ir.make_value(spv::OpAtomicSMax, ir.i64_type(), slot_ptr, /*scope=*/ir.const_i32_one_,
                  /*semantics=*/ir.const_i32_zero_, result_i64);

    ir.make_inst(spv::OpBranch, active_merge);
  }

  ir.start_label(early_return);
  ir.make_inst(spv::OpBranch, active_merge);

  ir.start_label(active_merge);
  ir.make_inst(spv::OpReturn);
  ir.make_inst(spv::OpFunctionEnd);

  std::vector<Value> entry_args = {args_buf, output_buf, params_buf, bytecode_buf};
  ir.commit_kernel_function(main_func, "main", entry_args, {static_cast<int>(kAdStackMaxReducerWorkgroupSize), 1, 1});

  return ir.finalize();
}

}  // namespace quadrants::lang::spirv
