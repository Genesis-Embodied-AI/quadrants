#include "quadrants/codegen/spirv/checkpoint_gate_shader.h"

#include "quadrants/codegen/spirv/spirv_ir_builder.h"

namespace quadrants::lang::spirv {

namespace {

// Read one u32 word from a storage-buffer-backed uint32[]. Mirrors the same-named helper in
// `adstack_bound_reducer_shader.cpp`; kept local so the gate shader's symbol set is self-contained.
Value load_buf_u32(IRBuilder &ir, Value buffer, Value word_idx) {
  Value ptr = ir.struct_array_access(ir.u32_type(), buffer, word_idx);
  return ir.load_variable(ptr, ir.u32_type());
}

// Write one u32 word into a storage-buffer-backed uint32[]. Plain `OpStore`; the gate runs as a
// single-thread workgroup so there's no cross-thread contention to fence against.
void store_buf_u32(IRBuilder &ir, Value buffer, Value word_idx, Value value) {
  Value ptr = ir.struct_array_access(ir.u32_type(), buffer, word_idx);
  ir.store_variable(ptr, value);
}

}  // namespace

std::vector<uint32_t> build_checkpoint_gate_spirv(Arch arch, const DeviceCapabilityConfig *caps) {
  IRBuilder ir(arch, caps);
  ir.init_header();

  // Bindings: see header doc.
  //   0: control { resume_point: i32, yield_signal: i32 }                -- viewed as u32[2]
  //   1: params  { cp_id: u32, n_kernels: u32, (gx,gy,gz) per kernel }   -- u32[2 + 3*N]
  //   2: out_dims { (gx,gy,gz) per kernel }                              -- u32[3*N]
  Value control_buf = ir.buffer_argument(ir.u32_type(), 0, 0, "checkpoint_gate_control");
  Value params_buf = ir.buffer_argument(ir.u32_type(), 0, 1, "checkpoint_gate_params");
  Value out_dims_buf = ir.buffer_argument(ir.u32_type(), 0, 2, "checkpoint_gate_out_dims");

  Value main_func = ir.new_function();
  ir.start_function(main_func);
  // 1x1x1: a single thread runs the per-kernel write loop. The total number of writes is at most
  // a few dozen per checkpoint (one (gx,gy,gz) triple per body kernel in the checkpoint); a
  // larger workgroup would just synchronise on the same loop with extra book-keeping.
  ir.set_work_group_size({1, 1, 1});

  // resume_point + yield_signal are int32 in semantics ("yield_signal == -1" means no yield), but
  // the SSBO is u32[]; reinterpret via `OpBitcast` for the comparisons. Same convention as the
  // CUDA-native `_qd_checkpoint_if_gate`.
  Value rp_u32 =
      load_buf_u32(ir, control_buf, ir.uint_immediate_number(ir.u32_type(), CheckpointControlBuf::kWordOffsetResumePoint));
  Value ys_u32 =
      load_buf_u32(ir, control_buf, ir.uint_immediate_number(ir.u32_type(), CheckpointControlBuf::kWordOffsetYieldSignal));
  Value cp_id_u32 =
      load_buf_u32(ir, params_buf, ir.uint_immediate_number(ir.u32_type(), CheckpointGateParams::kWordOffsetCpId));
  Value n_kernels_u32 =
      load_buf_u32(ir, params_buf, ir.uint_immediate_number(ir.u32_type(), CheckpointGateParams::kWordOffsetNKernels));

  Value rp_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), rp_u32);
  Value ys_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), ys_u32);
  Value cp_id_i32 = ir.make_value(spv::OpBitcast, ir.i32_type(), cp_id_u32);

  // skip := (cp_id < resume_point) || (yield_signal != -1)
  Value cp_below = ir.lt(cp_id_i32, rp_i32);
  Value neg_one = ir.int_immediate_number(ir.i32_type(), -1);
  Value yield_set = ir.ne(ys_i32, neg_one);
  Value skip = ir.make_value(spv::OpLogicalOr, ir.bool_type(), cp_below, yield_set);

  // Per-kernel write loop. SPIR-V structured control flow: header (decides continue), body
  // (writes the triple), continue (increments index), merge (loop exit).
  Value i_var = ir.alloca_variable(ir.u32_type());
  ir.store_variable(i_var, ir.uint_immediate_number(ir.u32_type(), 0u));

  Label head = ir.new_label();
  Label body = ir.new_label();
  Label cont = ir.new_label();
  Label merge = ir.new_label();

  ir.make_inst(spv::OpBranch, head);

  ir.start_label(head);
  Value i_now = ir.load_variable(i_var, ir.u32_type());
  Value loop_cond = ir.lt(i_now, n_kernels_u32);
  ir.make_inst(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
  ir.make_inst(spv::OpBranchConditional, loop_cond, body, merge);

  ir.start_label(body);
  {
    Value three = ir.uint_immediate_number(ir.u32_type(), 3u);
    Value dims_base = ir.uint_immediate_number(ir.u32_type(), CheckpointGateParams::kWordOffsetDimsBase);
    Value i3 = ir.mul(i_now, three);
    Value params_base = ir.add(dims_base, i3);
    Value out_base = i3;

    // For each of the three axes, load the active value from params and write either 0 or the
    // active value into out_dims, gated on `skip`. Use `OpSelect` (branch-free) since skip is a
    // single uniform value across the loop; it folds to a simple cmov on every backend.
    Value zero_u32 = ir.uint_immediate_number(ir.u32_type(), 0u);
    for (uint32_t axis = 0; axis < 3; ++axis) {
      Value off = ir.uint_immediate_number(ir.u32_type(), axis);
      Value params_idx = ir.add(params_base, off);
      Value out_idx = ir.add(out_base, off);
      Value active = load_buf_u32(ir, params_buf, params_idx);
      Value chosen = ir.select(skip, zero_u32, active);
      store_buf_u32(ir, out_dims_buf, out_idx, chosen);
    }
    ir.make_inst(spv::OpBranch, cont);
  }

  ir.start_label(cont);
  {
    Value one = ir.uint_immediate_number(ir.u32_type(), 1u);
    Value i_next = ir.add(i_now, one);
    ir.store_variable(i_var, i_next);
    ir.make_inst(spv::OpBranch, head);
  }

  ir.start_label(merge);
  ir.make_inst(spv::OpReturn);
  ir.make_inst(spv::OpFunctionEnd);

  std::vector<Value> entry_args = {control_buf, params_buf, out_dims_buf};
  ir.commit_kernel_function(main_func, "main", entry_args, {1, 1, 1});

  return ir.finalize();
}

}  // namespace quadrants::lang::spirv
