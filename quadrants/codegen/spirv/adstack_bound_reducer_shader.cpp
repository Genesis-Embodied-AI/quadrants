#include "quadrants/codegen/spirv/adstack_bound_reducer_shader.h"

#include "quadrants/codegen/spirv/spirv_ir_builder.h"

namespace quadrants::lang::spirv {

namespace {

// Small helper: read one uint32 word from a storage-buffer-backed uint32[] at the given scalar index. Mirrors the
// same-named helper in `adstack_sizer_shader.cpp`; kept local to this translation unit so the reducer's symbol set
// stays self-contained and the helper inlines without cross-file linkage.
Value load_buf_u32(IRBuilder &ir, Value buffer, Value word_idx) {
  Value ptr = ir.struct_array_access(ir.u32_type(), buffer, word_idx);
  return ir.load_variable(ptr, ir.u32_type());
}

// Assemble a u64 from two adjacent little-endian u32 words at `base_word_idx` and `base_word_idx + 1`. The kernel arg
// buffer's ndarray-pointer slot is laid out as two little-endian u32 words (the host launcher writes the u64 PSB
// pointer through a `memcpy` into the arg buffer); reading the two halves and reassembling matches the exact byte
// layout the main kernel sees when it consumes the same arg buffer. Returned as u64 (not bitcast to i64) because the
// only consumer is `OpConvertUToPtr` which takes an unsigned operand.
Value load_arg_buf_u64_ptr(IRBuilder &ir, Value buffer, Value base_word_idx) {
  Value lo = load_buf_u32(ir, buffer, base_word_idx);
  Value hi_idx = ir.add(base_word_idx, ir.uint_immediate_number(ir.u32_type(), 1u));
  Value hi = load_buf_u32(ir, buffer, hi_idx);
  Value lo64 = ir.cast(ir.u64_type(), lo);
  Value hi64 = ir.cast(ir.u64_type(), hi);
  Value shift = ir.uint_immediate_number(ir.u64_type(), 32u);
  Value hi_shifted = ir.make_value(spv::OpShiftLeftLogical, ir.u64_type(), hi64, shift);
  return ir.make_value(spv::OpBitwiseOr, ir.u64_type(), lo64, hi_shifted);
}

// Physical-Storage-Buffer load of one 32-bit scalar at byte offset `byte_off_u64` from `base_u64`. Mirrors the
// wrapper-struct PSB load pattern in `adstack_sizer_shader.cpp::psb_load_scalar`: `OpConvertUToPtr` to a
// pointer-to-wrapper-struct, then `OpAccessChain` on the `_m0` member, then `OpLoad` with the `Aligned` memory-access
// operand SPIR-V requires for `PhysicalStorageBuffer` reads. Caller passes the byte offset directly so the same helper
// covers the 4-byte-stride f32 / i32 walk and the 8-byte-stride f64 walk (issued as two adjacent 4-byte loads at
// offsets 0 and 4).
Value psb_load_u32_at_byte_off(IRBuilder &ir, Value base_u64, Value byte_off_u64) {
  Value target_u64 = ir.add(base_u64, byte_off_u64);

  SType elem_sty = ir.u32_type();
  SType ptr_elem_type = ir.get_pointer_type(elem_sty, spv::StorageClassPhysicalStorageBuffer);
  std::vector<std::tuple<SType, std::string, size_t>> members = {{elem_sty, "_m0", 0}};
  SType wrapper_struct = ir.create_struct_type(members);
  SType ptr_struct_type = ir.get_pointer_type(wrapper_struct, spv::StorageClassPhysicalStorageBuffer);
  Value struct_ptr = ir.make_value(spv::OpConvertUToPtr, ptr_struct_type, target_u64);
  Value scalar_ptr = ir.make_value(spv::OpAccessChain, ptr_elem_type, struct_ptr, ir.const_i32_zero_);
  Value scalar = ir.new_value(elem_sty, ValueKind::kNormal);
  ir.make_inst(spv::OpLoad, elem_sty, scalar, scalar_ptr, spv::MemoryAccessAlignedMask, /*alignment=*/4u);
  return scalar;
}

// Convenience wrapper around `psb_load_u32_at_byte_off` for the f32 / i32 path: byte offset is `elem_idx_u32 * 4`.
Value psb_load_u32(IRBuilder &ir, Value base_u64, Value elem_idx_u32) {
  Value four_u64 = ir.uint_immediate_number(ir.u64_type(), 4u);
  Value elem_idx_u64 = ir.cast(ir.u64_type(), elem_idx_u32);
  Value byte_off = ir.mul(elem_idx_u64, four_u64);
  return psb_load_u32_at_byte_off(ir, base_u64, byte_off);
}

// Assemble a u64 from two adjacent little-endian u32 PSB loads at byte offsets `elem_idx_u32 * 8` and `elem_idx_u32 * 8
// + 4`. PSB requires `Aligned 8` for a single 8-byte OpLoad; the source ndarray's element start is only guaranteed
// 4-byte aligned (it may follow a u32 in a containing struct), so we issue two 4-byte u32 loads and reassemble the u64
// in registers. The shifted-OR pattern mirrors `load_arg_buf_u64_ptr` above. Returned as u64 (not bitcast) because the
// caller does its own bitcast to f64 for the comparison.
Value psb_load_u64_pair(IRBuilder &ir, Value base_u64, Value elem_idx_u32) {
  Value eight_u64 = ir.uint_immediate_number(ir.u64_type(), 8u);
  Value four_u64 = ir.uint_immediate_number(ir.u64_type(), 4u);
  Value elem_idx_u64 = ir.cast(ir.u64_type(), elem_idx_u32);
  Value lo_byte_off = ir.mul(elem_idx_u64, eight_u64);
  Value hi_byte_off = ir.add(lo_byte_off, four_u64);
  Value lo = psb_load_u32_at_byte_off(ir, base_u64, lo_byte_off);
  Value hi = psb_load_u32_at_byte_off(ir, base_u64, hi_byte_off);
  Value lo64 = ir.cast(ir.u64_type(), lo);
  Value hi64 = ir.cast(ir.u64_type(), hi);
  Value shift = ir.uint_immediate_number(ir.u64_type(), 32u);
  Value hi_shifted = ir.make_value(spv::OpShiftLeftLogical, ir.u64_type(), hi64, shift);
  return ir.make_value(spv::OpBitwiseOr, ir.u64_type(), lo64, hi_shifted);
}

// Emits an i32 0/1 result for `lhs cmp rhs` with `cmp` selected by `op_code` at runtime via OpSwitch over the encoded
// `AdStackBoundReducerOpCode` values. The shader is generic, so `op_code` is loaded from the parameter blob rather than
// baked as a SpecConstant; the OpSwitch produces a tight straight-line dispatch in spirv-cross-emitted MSL on every
// `op_code` path. `is_float` switches between f32 and signed-i32 comparison; the SPIR-V comparison op codes for the two
// element kinds differ (FOrdLessThan vs SLessThan etc.), so we emit each kind in a separate branch.
Value emit_compare(IRBuilder &ir, Value lhs, Value rhs, Value op_code, bool is_float) {
  // Result is a u1 (bool). Each case emits the matching OpFOrd*/OpS* comparison; the default case (which should never
  // fire because the host clamps op_code to a valid `AdStackBoundReducerOpCode`) returns false to keep the per-thread
  // result well-defined.
  Label case_lt = ir.new_label();
  Label case_le = ir.new_label();
  Label case_gt = ir.new_label();
  Label case_ge = ir.new_label();
  Label case_eq = ir.new_label();
  Label case_ne = ir.new_label();
  Label case_default = ir.new_label();
  Label merge = ir.new_label();

  Value result_var = ir.alloca_variable(ir.bool_type());
  ir.store_variable(result_var, ir.uint_immediate_number(ir.bool_type(), 0u));

  ir.make_inst(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpSwitch, op_code, case_default, kAdStackBoundReducerOpLt, case_lt, kAdStackBoundReducerOpLe,
               case_le, kAdStackBoundReducerOpGt, case_gt, kAdStackBoundReducerOpGe, case_ge, kAdStackBoundReducerOpEq,
               case_eq, kAdStackBoundReducerOpNe, case_ne);

  auto store_cmp = [&](Label lbl, spv::Op f_op, spv::Op i_op) {
    ir.start_label(lbl);
    Value cmp = ir.new_value(ir.bool_type(), ValueKind::kNormal);
    ir.make_inst(is_float ? f_op : i_op, ir.bool_type(), cmp, lhs, rhs);
    ir.store_variable(result_var, cmp);
    ir.make_inst(spv::OpBranch, merge);
  };

  store_cmp(case_lt, spv::OpFOrdLessThan, spv::OpSLessThan);
  store_cmp(case_le, spv::OpFOrdLessThanEqual, spv::OpSLessThanEqual);
  store_cmp(case_gt, spv::OpFOrdGreaterThan, spv::OpSGreaterThan);
  store_cmp(case_ge, spv::OpFOrdGreaterThanEqual, spv::OpSGreaterThanEqual);
  store_cmp(case_eq, spv::OpFOrdEqual, spv::OpIEqual);
  store_cmp(case_ne, spv::OpFOrdNotEqual, spv::OpINotEqual);

  ir.start_label(case_default);
  ir.make_inst(spv::OpBranch, merge);

  ir.start_label(merge);
  return ir.load_variable(result_var, ir.bool_type());
}

}  // namespace

std::vector<uint32_t> build_adstack_bound_reducer_spirv(Arch arch, const DeviceCapabilityConfig *caps) {
  if (!caps->get(DeviceCapability::spirv_has_physical_storage_buffer)) {
    return {};
  }
  if (!caps->get(DeviceCapability::spirv_has_int64)) {
    return {};
  }

  IRBuilder ir(arch, caps);
  ir.init_header();

  // Storage-buffer bindings (set 0). Layout matches `AdStackBoundReducerParams` documentation in the header and the
  // host launcher's per-dispatch parameter-blob writeback path. All four are plain uint32[] arrays; `buffer_argument`
  // produces a SSBO-bound runtime array typed as u32 elements, and the per-thread loads index into them by word offset
  // (matching the encoder's little-endian POD-memcpy convention used for the arg buffer). Slot 3 is the root buffer for
  // SNode-backed gates - bound to the same root SSBO the main kernel uses, read at byte offset `snode_byte_base_offset
  // + gid * snode_byte_cell_stride` to load the gating field's value at cell `gid`. For ndarray-backed gates the host
  // can bind any non-null storage buffer here (the shader's load path against it is dead-stripped under spirv-opt's
  // branch elimination once `field_source_is_snode` is constant-folded by descriptor-set binding inputs).
  Value args_buf = ir.buffer_argument(ir.u32_type(), 0, 0, "adstack_bound_reducer_args");
  Value counter_buf = ir.buffer_argument(ir.u32_type(), 0, 1, "adstack_bound_reducer_counter");
  Value params_buf = ir.buffer_argument(ir.u32_type(), 0, 2, "adstack_bound_reducer_params");
  Value root_buf = ir.buffer_argument(ir.u32_type(), 0, 3, "adstack_bound_reducer_root");

  Value main_func = ir.new_function();
  ir.start_function(main_func);
  ir.set_work_group_size({static_cast<int>(kAdStackBoundReducerWorkgroupSize), 1, 1});

  // Per-thread invocation index. The host launcher dispatches `ceil(length / kWorkgroupSize)` workgroups, so `gid` may
  // exceed `length` on the trailing workgroup; the early-return below handles that case.
  Value gid_u32 = ir.get_global_invocation_id(0);

  // Load the parameter blob fields once at the top of `main`. spirv-opt CSEs the redundant param loads if they happen
  // multiple times within the same basic block, but keeping them explicit at the top makes the shader-side data-flow
  // easier to read.
  Value task_id = load_buf_u32(ir, params_buf,
                               ir.uint_immediate_number(ir.u32_type(), AdStackBoundReducerParams::kWordOffsetTaskId));
  Value length = load_buf_u32(ir, params_buf,
                              ir.uint_immediate_number(ir.u32_type(), AdStackBoundReducerParams::kWordOffsetLength));
  Value arg_word_offset = load_buf_u32(
      ir, params_buf, ir.uint_immediate_number(ir.u32_type(), AdStackBoundReducerParams::kWordOffsetArgWordOffset));
  Value op_code = load_buf_u32(ir, params_buf,
                               ir.uint_immediate_number(ir.u32_type(), AdStackBoundReducerParams::kWordOffsetOpCode));
  Value field_dtype_is_float_u32 = load_buf_u32(
      ir, params_buf, ir.uint_immediate_number(ir.u32_type(), AdStackBoundReducerParams::kWordOffsetFieldDtypeIsFloat));
  Value polarity_u32 = load_buf_u32(
      ir, params_buf, ir.uint_immediate_number(ir.u32_type(), AdStackBoundReducerParams::kWordOffsetPolarity));
  Value threshold_bits = load_buf_u32(
      ir, params_buf, ir.uint_immediate_number(ir.u32_type(), AdStackBoundReducerParams::kWordOffsetThresholdBits));
  Value field_source_is_snode_u32 =
      load_buf_u32(ir, params_buf,
                   ir.uint_immediate_number(ir.u32_type(), AdStackBoundReducerParams::kWordOffsetFieldSourceIsSnode));
  Value snode_byte_base_offset =
      load_buf_u32(ir, params_buf,
                   ir.uint_immediate_number(ir.u32_type(), AdStackBoundReducerParams::kWordOffsetSnodeByteBaseOffset));
  Value snode_byte_cell_stride =
      load_buf_u32(ir, params_buf,
                   ir.uint_immediate_number(ir.u32_type(), AdStackBoundReducerParams::kWordOffsetSnodeByteCellStride));
  // f64 gate extension. Only consulted when the device supports `spirv_has_float64`; on devices without f64 the host
  // launcher filters f64-captured bound_exprs out of the dispatch (falling back to worst-case heap sizing) and these
  // slots are never read. Loading them unconditionally keeps the shader's static layout matched against the host
  // launcher's params blob writeback.
  const bool has_f64 = caps->get(DeviceCapability::spirv_has_float64);
  Value field_dtype_is_double_u32 =
      load_buf_u32(ir, params_buf,
                   ir.uint_immediate_number(ir.u32_type(), AdStackBoundReducerParams::kWordOffsetFieldDtypeIsDouble));
  Value threshold_bits_high = load_buf_u32(
      ir, params_buf, ir.uint_immediate_number(ir.u32_type(), AdStackBoundReducerParams::kWordOffsetThresholdBitsHigh));

  // Trailing-workgroup bounds check. `gid >= length` threads exit early; remaining threads atomic-add into the counter
  // slot. The early return must be a structured branch so spirv-val accepts the function body (SPIR-V 1.0
  // selection-merge rules).
  Label active_block = ir.new_label();
  Label early_return = ir.new_label();
  Label active_merge = ir.new_label();
  Value in_range = ir.lt(gid_u32, length);
  ir.make_inst(spv::OpSelectionMerge, active_merge, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpBranchConditional, in_range, active_block, early_return);

  ir.start_label(active_block);
  {
    // Resolve the gating field's element at `gid`. Two source kinds are supported, branched on
    // `field_source_is_snode_u32`: ndarray-backed (read the data pointer out of the kernel arg buffer at the
    // encoder-precomputed word offset, PSB-load the element at `gid`) and SNode-backed (compute byte offset
    // `snode_byte_base_offset + gid * snode_byte_cell_stride` directly into the bound root buffer and load the element
    // word(s)). The element width is 4 bytes for f32 / i32 and 8 bytes for f64; the f64 path issues two adjacent 4-byte
    // loads and reassembles into a u64. We always materialise the loaded value as a u64 (low 32 bits zero-extended in
    // the f32 / i32 case) so the dtype-branch downstream can pick f64 / f32 / i32 reinterpretation without re-loading.
    Value field_u64_var = ir.alloca_variable(ir.u64_type());
    Value is_double = ir.ne(field_dtype_is_double_u32, ir.uint_immediate_number(ir.u32_type(), 0u));
    Value field_source_is_snode = ir.ne(field_source_is_snode_u32, ir.uint_immediate_number(ir.u32_type(), 0u));
    Label src_snode_lbl = ir.new_label();
    Label src_ndarr_lbl = ir.new_label();
    Label src_merge = ir.new_label();
    ir.make_inst(spv::OpSelectionMerge, src_merge, spv::SelectionControlMaskNone);
    ir.make_inst(spv::OpBranchConditional, field_source_is_snode, src_snode_lbl, src_ndarr_lbl);

    ir.start_label(src_snode_lbl);
    {
      // SNode root buffer is a u32[] view. Per-snode-descriptor alignment guarantees `snode_byte_base_offset` and
      // `snode_byte_cell_stride` are multiples of 4. f32 / i32 fields walk one 4-byte word per cell; f64 fields walk
      // two adjacent 4-byte words and reassemble into a u64. Issuing two u32 loads (rather than one u64 load) keeps the
      // alignment requirement at 4 bytes so any dense parent's f64-cell layout works without further alignment
      // promotion in the descriptor binding.
      Value byte_off = ir.add(snode_byte_base_offset, ir.mul(gid_u32, snode_byte_cell_stride));
      Value lo_word_idx = ir.div(byte_off, ir.uint_immediate_number(ir.u32_type(), 4u));
      Value lo = load_buf_u32(ir, root_buf, lo_word_idx);
      Label snode_dbl_lbl = ir.new_label();
      Label snode_sgl_lbl = ir.new_label();
      Label snode_merge = ir.new_label();
      ir.make_inst(spv::OpSelectionMerge, snode_merge, spv::SelectionControlMaskNone);
      ir.make_inst(spv::OpBranchConditional, is_double, snode_dbl_lbl, snode_sgl_lbl);

      ir.start_label(snode_dbl_lbl);
      {
        Value hi_word_idx = ir.add(lo_word_idx, ir.uint_immediate_number(ir.u32_type(), 1u));
        Value hi = load_buf_u32(ir, root_buf, hi_word_idx);
        Value lo64 = ir.cast(ir.u64_type(), lo);
        Value hi64 = ir.cast(ir.u64_type(), hi);
        Value shift = ir.uint_immediate_number(ir.u64_type(), 32u);
        Value hi_shifted = ir.make_value(spv::OpShiftLeftLogical, ir.u64_type(), hi64, shift);
        Value combined = ir.make_value(spv::OpBitwiseOr, ir.u64_type(), lo64, hi_shifted);
        ir.store_variable(field_u64_var, combined);
        ir.make_inst(spv::OpBranch, snode_merge);
      }
      ir.start_label(snode_sgl_lbl);
      {
        ir.store_variable(field_u64_var, ir.cast(ir.u64_type(), lo));
        ir.make_inst(spv::OpBranch, snode_merge);
      }
      ir.start_label(snode_merge);
      ir.make_inst(spv::OpBranch, src_merge);
    }

    ir.start_label(src_ndarr_lbl);
    {
      // ndarray-backed: PSB-load one u32 (f32 / i32) or two adjacent u32 words (f64). The base pointer is assembled
      // from the two arg-buffer u32 words at `arg_word_offset` and `arg_word_offset + 1`.
      Value ndarray_ptr_u64 = load_arg_buf_u64_ptr(ir, args_buf, arg_word_offset);
      Label ndarr_dbl_lbl = ir.new_label();
      Label ndarr_sgl_lbl = ir.new_label();
      Label ndarr_merge = ir.new_label();
      ir.make_inst(spv::OpSelectionMerge, ndarr_merge, spv::SelectionControlMaskNone);
      ir.make_inst(spv::OpBranchConditional, is_double, ndarr_dbl_lbl, ndarr_sgl_lbl);

      ir.start_label(ndarr_dbl_lbl);
      {
        Value combined = psb_load_u64_pair(ir, ndarray_ptr_u64, gid_u32);
        ir.store_variable(field_u64_var, combined);
        ir.make_inst(spv::OpBranch, ndarr_merge);
      }
      ir.start_label(ndarr_sgl_lbl);
      {
        Value loaded = psb_load_u32(ir, ndarray_ptr_u64, gid_u32);
        ir.store_variable(field_u64_var, ir.cast(ir.u64_type(), loaded));
        ir.make_inst(spv::OpBranch, ndarr_merge);
      }
      ir.start_label(ndarr_merge);
      ir.make_inst(spv::OpBranch, src_merge);
    }

    ir.start_label(src_merge);
    Value field_u64 = ir.load_variable(field_u64_var, ir.u64_type());

    // Branch on `field_dtype_is_float`. The float path further forks on `is_double` to pick f32 vs f64 bitcast +
    // comparison; the int path (i32) truncates the u64 to u32 and bitcasts to i32. The f64 inner arm is only emitted on
    // devices advertising `spirv_has_float64`; on f32-only devices the host launcher filters f64-captured bound_exprs
    // out of the dispatch entirely (see adstack_bound_reducer_launch.cpp's matched-task filter), so the inner arm is
    // never reached at runtime - and skipping its emission keeps the OpType for f64 out of the binary, which is what
    // spirv-val requires on f32-only targets.
    Label float_lbl = ir.new_label();
    Label int_lbl = ir.new_label();
    Label dtype_merge = ir.new_label();

    // See note above: `alloca_variable` hoists OpVariable to the entry block; pair with stores on every reachable path
    // through the dtype-branch so the merge-block load never sees undef.
    Value matched_var = ir.alloca_variable(ir.bool_type());
    Value is_float = ir.ne(field_dtype_is_float_u32, ir.uint_immediate_number(ir.u32_type(), 0u));
    ir.make_inst(spv::OpSelectionMerge, dtype_merge, spv::SelectionControlMaskNone);
    ir.make_inst(spv::OpBranchConditional, is_float, float_lbl, int_lbl);

    ir.start_label(float_lbl);
    {
      if (has_f64) {
        Label f64_lbl = ir.new_label();
        Label f32_lbl = ir.new_label();
        Label float_inner_merge = ir.new_label();
        ir.make_inst(spv::OpSelectionMerge, float_inner_merge, spv::SelectionControlMaskNone);
        ir.make_inst(spv::OpBranchConditional, is_double, f64_lbl, f32_lbl);

        ir.start_label(f64_lbl);
        {
          Value field_d = ir.make_value(spv::OpBitcast, ir.f64_type(), field_u64);
          Value lo64 = ir.cast(ir.u64_type(), threshold_bits);
          Value hi64 = ir.cast(ir.u64_type(), threshold_bits_high);
          Value shift = ir.uint_immediate_number(ir.u64_type(), 32u);
          Value hi_shifted = ir.make_value(spv::OpShiftLeftLogical, ir.u64_type(), hi64, shift);
          Value threshold_u64 = ir.make_value(spv::OpBitwiseOr, ir.u64_type(), lo64, hi_shifted);
          Value threshold_d = ir.make_value(spv::OpBitcast, ir.f64_type(), threshold_u64);
          Value cmp = emit_compare(ir, field_d, threshold_d, op_code, /*is_float=*/true);
          ir.store_variable(matched_var, cmp);
          ir.make_inst(spv::OpBranch, float_inner_merge);
        }
        ir.start_label(f32_lbl);
        {
          Value field_word = ir.cast(ir.u32_type(), field_u64);
          Value field_f = ir.make_value(spv::OpBitcast, ir.f32_type(), field_word);
          Value threshold_f = ir.make_value(spv::OpBitcast, ir.f32_type(), threshold_bits);
          Value cmp = emit_compare(ir, field_f, threshold_f, op_code, /*is_float=*/true);
          ir.store_variable(matched_var, cmp);
          ir.make_inst(spv::OpBranch, float_inner_merge);
        }
        ir.start_label(float_inner_merge);
      } else {
        Value field_word = ir.cast(ir.u32_type(), field_u64);
        Value field_f = ir.make_value(spv::OpBitcast, ir.f32_type(), field_word);
        Value threshold_f = ir.make_value(spv::OpBitcast, ir.f32_type(), threshold_bits);
        Value cmp = emit_compare(ir, field_f, threshold_f, op_code, /*is_float=*/true);
        ir.store_variable(matched_var, cmp);
      }
      ir.make_inst(spv::OpBranch, dtype_merge);
    }

    ir.start_label(int_lbl);
    {
      Value field_word = ir.cast(ir.u32_type(), field_u64);
      Value field_i = ir.make_value(spv::OpBitcast, ir.i32_type(), field_word);
      Value threshold_i = ir.make_value(spv::OpBitcast, ir.i32_type(), threshold_bits);
      Value cmp = emit_compare(ir, field_i, threshold_i, op_code, /*is_float=*/false);
      ir.store_variable(matched_var, cmp);
      ir.make_inst(spv::OpBranch, dtype_merge);
    }

    ir.start_label(dtype_merge);
    Value matched = ir.load_variable(matched_var, ir.bool_type());

    // Apply polarity. The captured `StaticBoundExpr::polarity` is true when the LCA enters on the predicate holding
    // (typical `if cmp:` shape) and false when the LCA sits inside the `else` branch; in the latter case the count we
    // want is "threads where the predicate is FALSE", so we XOR-flip with `!polarity`.
    Value polarity_u1 = ir.ne(polarity_u32, ir.uint_immediate_number(ir.u32_type(), 0u));
    Value not_polarity = ir.make_value(spv::OpLogicalNot, ir.bool_type(), polarity_u1);
    Value should_count = ir.make_value(spv::OpLogicalNotEqual, ir.bool_type(), matched, not_polarity);

    Label count_block = ir.new_label();
    Label count_merge = ir.new_label();
    ir.make_inst(spv::OpSelectionMerge, count_merge, spv::SelectionControlMaskNone);
    ir.make_inst(spv::OpBranchConditional, should_count, count_block, count_merge);

    ir.start_label(count_block);
    {
      // Atomic-add 1 into `counter_buf[task_id]`. Memory scope = Device, semantics = Relaxed (the captured count is
      // consumed by the host post-`wait_idle`, so the kernel does not require an in-shader fence).
      Value slot_ptr = ir.struct_array_access(ir.u32_type(), counter_buf, task_id);
      ir.make_value(spv::OpAtomicIAdd, ir.u32_type(), slot_ptr, /*scope=*/ir.const_i32_one_,
                    /*semantics=*/ir.const_i32_zero_, ir.uint_immediate_number(ir.u32_type(), 1u));
      ir.make_inst(spv::OpBranch, count_merge);
    }

    ir.start_label(count_merge);
    ir.make_inst(spv::OpBranch, active_merge);
  }

  ir.start_label(early_return);
  ir.make_inst(spv::OpBranch, active_merge);

  ir.start_label(active_merge);
  ir.make_inst(spv::OpReturn);
  ir.make_inst(spv::OpFunctionEnd);

  std::vector<Value> entry_args = {args_buf, counter_buf, params_buf, root_buf};
  ir.commit_kernel_function(main_func, "main", entry_args, {static_cast<int>(kAdStackBoundReducerWorkgroupSize), 1, 1});

  return ir.finalize();
}

}  // namespace quadrants::lang::spirv
