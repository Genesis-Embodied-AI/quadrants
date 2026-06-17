#include "quadrants/codegen/spirv/checkpoint_yield_check_shader.h"

#include "quadrants/codegen/spirv/checkpoint_gate_shader.h"
#include "quadrants/codegen/spirv/spirv_ir_builder.h"

namespace quadrants::lang::spirv {

namespace {

Value load_buf_u32(IRBuilder &ir, Value buffer, Value word_idx) {
  Value ptr = ir.struct_array_access(ir.u32_type(), buffer, word_idx);
  return ir.load_variable(ptr, ir.u32_type());
}

}  // namespace

std::vector<uint32_t> build_checkpoint_yield_check_spirv(Arch arch, const DeviceCapabilityConfig *caps) {
  IRBuilder ir(arch, caps);
  ir.init_header();

  // Bindings: see header doc.
  Value control_buf = ir.buffer_argument(ir.u32_type(), 0, 0, "checkpoint_yc_control");
  Value yield_on_buf = ir.buffer_argument(ir.u32_type(), 0, 1, "checkpoint_yc_yield_on");
  Value params_buf = ir.buffer_argument(ir.u32_type(), 0, 2, "checkpoint_yc_params");

  Value main_func = ir.new_function();
  ir.start_function(main_func);
  ir.set_work_group_size({1, 1, 1});

  // Read flag once. The CUDA-native yield-check uses a non-atomic `*yield_on` load because the checkpoint body's writes
  // were already serialised by the graph node dependency; on Vulkan / Metal the cmdlist's between-dispatch
  // `memory_barrier()` call provides the same ordering, so a plain load is also sufficient here.
  Value flag = load_buf_u32(ir, yield_on_buf, ir.uint_immediate_number(ir.u32_type(), 0u));
  Value zero_u32 = ir.uint_immediate_number(ir.u32_type(), 0u);
  Value flag_set = ir.ne(flag, zero_u32);

  Label body_lbl = ir.new_label();
  Label merge_lbl = ir.new_label();
  ir.make_inst(spv::OpSelectionMerge, merge_lbl, spv::SelectionControlMaskNone);
  ir.make_inst(spv::OpBranchConditional, flag_set, body_lbl, merge_lbl);

  ir.start_label(body_lbl);
  {
    // atomicCAS(yield_signal, -1, cp_id). First-yielder-wins: if some earlier checkpoint already raised yield this
    // launch, that earlier cp_id stays in the slot; later checkpoints see `old != -1` and noop the swap. Matches
    // `checkpoint_yield_check.cu`'s `atomicCAS`.
    Value cp_id_u32 = load_buf_u32(
        ir, params_buf, ir.uint_immediate_number(ir.u32_type(), CheckpointYieldCheckParams::kWordOffsetCpId));

    Value ys_ptr =
        ir.struct_array_access(ir.u32_type(), control_buf,
                               ir.uint_immediate_number(ir.u32_type(), CheckpointControlBuf::kWordOffsetYieldSignal));
    Value neg_one_u32 = ir.uint_immediate_number(ir.u32_type(), 0xFFFFFFFFu);  // -1 as u32 bit pattern
    // OpAtomicCompareExchange returns the previous value; we don't consume it. Scope = Device (`1`), Semantics =
    // Relaxed (`0`); the surrounding cmdlist barriers carry the ordering.
    ir.make_value(spv::OpAtomicCompareExchange, ir.u32_type(), ys_ptr,
                  /*scope=*/ir.const_i32_one_, /*sem_eq=*/ir.const_i32_zero_,
                  /*sem_neq=*/ir.const_i32_zero_, /*value=*/cp_id_u32, /*comparator=*/neg_one_u32);

    // We deliberately do NOT clear the user's `yield_on` buffer here -- the framework never writes into user-owned
    // ndarrays. The host loop must clear it before the resume launch (see `docs/source/user_guide/graph.md`).
    ir.make_inst(spv::OpBranch, merge_lbl);
  }

  ir.start_label(merge_lbl);
  ir.make_inst(spv::OpReturn);
  ir.make_inst(spv::OpFunctionEnd);

  std::vector<Value> entry_args = {control_buf, yield_on_buf, params_buf};
  ir.commit_kernel_function(main_func, "main", entry_args, {1, 1, 1});

  return ir.finalize();
}

}  // namespace quadrants::lang::spirv
