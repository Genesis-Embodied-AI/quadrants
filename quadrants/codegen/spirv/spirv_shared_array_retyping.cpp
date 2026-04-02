// Note: this module operates only on Quadrants IR types (AllocaStmt, Block,
// etc.) and SPIR-V types — no LLVM types are involved.

#include "quadrants/codegen/spirv/spirv_shared_array_retyping.h"

#include "quadrants/ir/type_utils.h"

namespace quadrants::lang {
namespace spirv {
namespace {

// Follow MatrixPtrStmt::origin chains back to the source AllocaStmt.
// Assumes only MatrixPtrStmt's on the chain to get to the AllocStmt.
const AllocaStmt *trace_to_alloca(const Stmt *stmt) {
  if (auto *alloca = stmt->cast<AllocaStmt>())
    return alloca;
  if (auto *matrix_ptr = stmt->cast<MatrixPtrStmt>())
    return trace_to_alloca(matrix_ptr->origin);
  return nullptr;
}

DataType get_atomic_uint_dtype(IRBuilder &ir, const DataType &dt) {
  DataType uint_dt = ir.get_quadrants_uint_type(dt);
  if (uint_dt == PrimitiveType::u16 || uint_dt == PrimitiveType::u8) {
    return PrimitiveType::u32;
  }
  return uint_dt;
}

// CAS loop with width-aware uint<->float conversion. The atomic backing type
// (res_type) may be wider than the float type (e.g. u32 for f16), so
// OpUConvert narrows/widens around the bitcasts.
Value atomic_operation_widened(IRBuilder &ir,
                               Value addr_ptr,
                               Value data,
                               std::function<Value(Value, Value)> op,
                               const DataType &dt,
                               const DataType &atomic_uint_dt) {
  SType float_type = ir.get_primitive_type(dt);
  SType narrow_uint = ir.get_primitive_uint_type(dt);
  SType res_type = ir.get_primitive_type(atomic_uint_dt);
  Value ret_val_int = ir.alloca_variable(res_type);

  Label head = ir.new_label();
  Label body = ir.new_label();
  Label branch_true = ir.new_label();
  Label branch_false = ir.new_label();
  Label merge = ir.new_label();
  Label exit = ir.new_label();

  ir.make_inst(spv::OpBranch, head);
  ir.start_label(head);
  ir.make_inst(spv::OpLoopMerge, branch_true, merge, 0);
  ir.make_inst(spv::OpBranch, body);
  ir.make_inst(spv::OpLabel, body);
  {
    // See IRBuilder::atomic_operation for why OpAtomicLoad is used here
    // instead of OpLoad (prevents SPIRV-Cross inlining on Metal).
    Value old_val = ir.make_value(spv::OpAtomicLoad, res_type, addr_ptr,
                                  /*scope=*/ir.const_i32_one_,
                                  /*semantics=*/ir.const_i32_zero_);
    // uint -> float (narrowing if atomic type is wider)
    Value old_narrow = old_val;
    if (res_type.id != narrow_uint.id) {
      old_narrow = ir.make_value(spv::OpUConvert, narrow_uint, old_val);
    }
    Value old_data_value =
        ir.make_value(spv::OpBitcast, float_type, old_narrow);
    Value new_data_value = op(old_data_value, data);
    // float -> uint (widening if needed)
    Value new_val = ir.make_value(spv::OpBitcast, narrow_uint, new_data_value);
    if (res_type.id != narrow_uint.id) {
      new_val = ir.make_value(spv::OpUConvert, res_type, new_val);
    }
    Value loaded = ir.make_value(
        spv::OpAtomicCompareExchange, res_type, addr_ptr,
        /*scope=*/ir.const_i32_one_, /*semantics if equal=*/ir.const_i32_zero_,
        /*semantics if unequal=*/ir.const_i32_zero_, new_val, old_val);
    Value ok = ir.make_value(spv::OpIEqual, ir.bool_type(), loaded, old_val);
    ir.store_variable(ret_val_int, loaded);
    ir.make_inst(spv::OpSelectionMerge, branch_false, 0);
    ir.make_inst(spv::OpBranchConditional, ok, branch_true, branch_false);
    {
      ir.make_inst(spv::OpLabel, branch_true);
      ir.make_inst(spv::OpBranch, exit);
    }
    {
      ir.make_inst(spv::OpLabel, branch_false);
      ir.make_inst(spv::OpBranch, merge);
    }
    ir.make_inst(spv::OpLabel, merge);
    ir.make_inst(spv::OpBranch, head);
  }
  ir.start_label(exit);

  Value ret_loaded = ir.load_variable(ret_val_int, res_type);
  if (res_type.id != narrow_uint.id) {
    ret_loaded = ir.make_value(spv::OpUConvert, narrow_uint, ret_loaded);
  }
  return ir.make_value(spv::OpBitcast, float_type, ret_loaded);
}

}  // namespace

void scan_shared_atomic_allocs(Block *ir_block,
                               std::unordered_map<const Stmt *, bool> &out) {
  for (auto &s : ir_block->statements) {
    if (auto *atomic_stmt = s->cast<AtomicOpStmt>()) {
      if (auto *alloca = trace_to_alloca(atomic_stmt->dest)) {
        if (alloca->is_shared) {
          // alloca->ret_type is a pointer to the stored type;
          // ptr_removed() gives the stored type (e.g. array of 128 floats).
          auto alloca_dtype = alloca->ret_type.ptr_removed();
          // Shared array is always modeled as a tensor type.
          if (auto *tensor_type = alloca_dtype->cast<TensorType>()) {
            auto scalar_dtype = tensor_type->get_element_type();
            if (auto *nested = scalar_dtype->cast<TensorType>()) {
              scalar_dtype = nested->get_element_type();
              QD_ASSERT_INFO(
                  !scalar_dtype->cast<TensorType>(),
                  "Nested tensor types deeper than 2 levels not supported");
            }
            if (is_real(scalar_dtype)) {
              bool has_non_add = (atomic_stmt->op_type != AtomicOpType::add);
              auto [it, inserted] = out.emplace(alloca, has_non_add);
              if (!inserted)
                it->second = it->second || has_non_add;
            }
          }
        }
      }
    }
    // Recurse into sub-blocks.
    // StructForStmt and MeshForStmt are lowered before codegen.
    QD_ASSERT(!s->cast<StructForStmt>());
    QD_ASSERT(!s->cast<MeshForStmt>());
    if (auto *if_stmt = s->cast<IfStmt>()) {
      if (if_stmt->true_statements)
        scan_shared_atomic_allocs(if_stmt->true_statements.get(), out);
      if (if_stmt->false_statements)
        scan_shared_atomic_allocs(if_stmt->false_statements.get(), out);
    } else if (auto *range_for = s->cast<RangeForStmt>()) {
      scan_shared_atomic_allocs(range_for->body.get(), out);
    } else if (auto *while_stmt = s->cast<WhileStmt>()) {
      scan_shared_atomic_allocs(while_stmt->body.get(), out);
    }
  }
}

SType maybe_retype_shared_float_alloca(
    IRBuilder &ir,
    const DeviceCapabilityConfig &caps,
    const AllocaStmt *alloca,
    const DataType &scalar_dtype,
    const std::unordered_map<const Stmt *, bool> &alloc_map,
    std::unordered_set<const Stmt *> &retyped_stmts) {
  auto stype = ir.get_primitive_type(scalar_dtype);
  auto it = alloc_map.find(alloca);
  if (it == alloc_map.end())
    return stype;
  bool needs_cas = it->second;
  if (!needs_cas && has_native_float_atomic_add(caps, scalar_dtype, true))
    return stype;
  stype = ir.get_primitive_type(get_atomic_uint_dtype(ir, scalar_dtype));
  retyped_stmts.insert(alloca);
  return stype;
}

SType maybe_retype_derived_ptr(
    IRBuilder &ir,
    const Stmt *origin,
    const Stmt *stmt,
    const DataType &dt,
    std::unordered_set<const Stmt *> &retyped_stmts) {
  auto stype = ir.get_primitive_type(dt);
  if (retyped_stmts.count(origin)) {
    stype = ir.get_primitive_type(get_atomic_uint_dtype(ir, dt));
    retyped_stmts.insert(stmt);
  }
  return stype;
}

Value load_shared_float(IRBuilder &ir,
                        Value ptr_val,
                        const Stmt *ptr,
                        const DataType &element_type,
                        const std::unordered_set<const Stmt *> &retyped_stmts) {
  if (retyped_stmts.count(ptr)) {
    auto shared_type =
        ir.get_primitive_type(get_atomic_uint_dtype(ir, element_type));
    Value val = ir.load_variable(ptr_val, shared_type);
    SType narrow_uint = ir.get_primitive_uint_type(element_type);
    if (shared_type.id != narrow_uint.id) {
      val = ir.make_value(spv::OpUConvert, narrow_uint, val);
    }
    return ir.make_value(spv::OpBitcast, ir.get_primitive_type(element_type),
                         val);
  }
  return ir.load_variable(ptr_val, ir.get_primitive_type(element_type));
}

Value store_shared_float(
    IRBuilder &ir,
    Value val,
    const Stmt *dest,
    const DataType &val_type,
    const std::unordered_set<const Stmt *> &retyped_stmts) {
  if (retyped_stmts.count(dest)) {
    SType narrow_uint = ir.get_primitive_uint_type(val_type);
    val = ir.make_value(spv::OpBitcast, narrow_uint, val);
    SType atomic_uint =
        ir.get_primitive_type(get_atomic_uint_dtype(ir, val_type));
    if (atomic_uint.id != narrow_uint.id) {
      val = ir.make_value(spv::OpUConvert, atomic_uint, val);
    }
  }
  return val;
}

Value shared_float_atomic(IRBuilder &ir,
                          AtomicOpType op_type,
                          Value addr_ptr,
                          Value data,
                          const DataType &dt) {
  auto atomic_uint_dt = get_atomic_uint_dtype(ir, dt);
  auto float_type = ir.get_primitive_type(dt);
  if (op_type == AtomicOpType::add) {
    return atomic_operation_widened(
        ir, addr_ptr, data,
        [&](Value lhs, Value rhs) { return ir.add(lhs, rhs); }, dt,
        atomic_uint_dt);
  } else if (op_type == AtomicOpType::sub) {
    return atomic_operation_widened(
        ir, addr_ptr, data,
        [&](Value lhs, Value rhs) { return ir.sub(lhs, rhs); }, dt,
        atomic_uint_dt);
  } else if (op_type == AtomicOpType::mul) {
    return atomic_operation_widened(
        ir, addr_ptr, data,
        [&](Value lhs, Value rhs) { return ir.mul(lhs, rhs); }, dt,
        atomic_uint_dt);
  } else if (op_type == AtomicOpType::min) {
    return atomic_operation_widened(
        ir, addr_ptr, data,
        [&](Value lhs, Value rhs) {
          return ir.call_glsl450(float_type, /*FMin*/ 37, lhs, rhs);
        },
        dt, atomic_uint_dt);
  } else if (op_type == AtomicOpType::max) {
    return atomic_operation_widened(
        ir, addr_ptr, data,
        [&](Value lhs, Value rhs) {
          return ir.call_glsl450(float_type, /*FMax*/ 40, lhs, rhs);
        },
        dt, atomic_uint_dt);
  } else {
    QD_NOT_IMPLEMENTED
  }
}

bool has_native_float_atomic_add(const DeviceCapabilityConfig &caps,
                                 const DataType &dt,
                                 bool is_shared) {
  if (dt->is_primitive(PrimitiveTypeID::f32))
    return caps.get(is_shared
                        ? DeviceCapability::spirv_has_shared_atomic_float_add
                        : DeviceCapability::spirv_has_atomic_float_add);
  if (dt->is_primitive(PrimitiveTypeID::f64))
    return caps.get(is_shared
                        ? DeviceCapability::spirv_has_shared_atomic_float64_add
                        : DeviceCapability::spirv_has_atomic_float64_add);
  if (dt->is_primitive(PrimitiveTypeID::f16))
    return caps.get(is_shared
                        ? DeviceCapability::spirv_has_shared_atomic_float16_add
                        : DeviceCapability::spirv_has_atomic_float16_add);
  return false;
}

}  // namespace spirv
}  // namespace quadrants::lang
