#include "quadrants/codegen/spirv/spirv_operations.h"

namespace quadrants::lang::spirv {

Value SpirvOperations::popcnt(Value x) {
  QD_ASSERT(is_integral(x.stype.dt));
  return ir_.make_value(spv::OpBitCount, x.stype, x);
}

#define DEFINE_SPIRV_BINARY_USIGN_OP(_OpName, _Op)         \
  Value SpirvOperations::_OpName(Value a, Value b) {       \
    QD_ASSERT(a.stype.id == b.stype.id);                   \
    if (is_integral(a.stype.dt)) {                         \
      return ir_.make_value(spv::OpI##_Op, a.stype, a, b); \
    } else {                                               \
      QD_ASSERT(is_real(a.stype.dt));                      \
      return ir_.make_value(spv::OpF##_Op, a.stype, a, b); \
    }                                                      \
  }

#define DEFINE_SPIRV_BINARY_SIGN_OP(_OpName, _Op)           \
  Value SpirvOperations::_OpName(Value a, Value b) {        \
    QD_ASSERT(a.stype.id == b.stype.id);                    \
    if (is_integral(a.stype.dt) && is_signed(a.stype.dt)) { \
      return ir_.make_value(spv::OpS##_Op, a.stype, a, b);  \
    } else if (is_integral(a.stype.dt)) {                   \
      return ir_.make_value(spv::OpU##_Op, a.stype, a, b);  \
    } else {                                                \
      QD_ASSERT(is_real(a.stype.dt));                       \
      return ir_.make_value(spv::OpF##_Op, a.stype, a, b);  \
    }                                                       \
  }

DEFINE_SPIRV_BINARY_USIGN_OP(add, Add);
DEFINE_SPIRV_BINARY_USIGN_OP(sub, Sub);
DEFINE_SPIRV_BINARY_USIGN_OP(mul, Mul);
DEFINE_SPIRV_BINARY_SIGN_OP(div, Div);

Value SpirvOperations::mod(Value a, Value b) {
  QD_ASSERT(a.stype.id == b.stype.id);
  if (is_integral(a.stype.dt) && is_signed(a.stype.dt)) {
    // FIXME: figure out why OpSRem does not work
    return sub(a, mul(b, div(a, b)));
  } else if (is_integral(a.stype.dt)) {
    return ir_.make_value(spv::OpUMod, a.stype, a, b);
  } else {
    QD_ASSERT(is_real(a.stype.dt));
    return ir_.make_value(spv::OpFRem, a.stype, a, b);
  }
}

#define DEFINE_SPIRV_CMP_OP(_OpName, _Op)                                          \
  Value SpirvOperations::_OpName(Value a, Value b) {                               \
    QD_ASSERT(a.stype.id == b.stype.id);                                           \
    const auto &bool_type = ir_.bool_type(); /* TODO: Only scalar supported now */ \
    if (is_integral(a.stype.dt) && is_signed(a.stype.dt)) {                        \
      return ir_.make_value(spv::OpS##_Op, bool_type, a, b);                       \
    } else if (is_integral(a.stype.dt)) {                                          \
      return ir_.make_value(spv::OpU##_Op, bool_type, a, b);                       \
    } else {                                                                       \
      QD_ASSERT(is_real(a.stype.dt));                                              \
      return ir_.make_value(spv::OpFOrd##_Op, bool_type, a, b);                    \
    }                                                                              \
  }

DEFINE_SPIRV_CMP_OP(lt, LessThan);
DEFINE_SPIRV_CMP_OP(le, LessThanEqual);
DEFINE_SPIRV_CMP_OP(gt, GreaterThan);
DEFINE_SPIRV_CMP_OP(ge, GreaterThanEqual);

#define DEFINE_SPIRV_CMP_UOP(_OpName, _Op)                                         \
  Value SpirvOperations::_OpName(Value a, Value b) {                               \
    QD_ASSERT(a.stype.id == b.stype.id);                                           \
    const auto &bool_type = ir_.bool_type(); /* TODO: Only scalar supported now */ \
    if (a.stype.id == bool_type.id) {                                              \
      return ir_.make_value(spv::OpLogical##_Op, bool_type, a, b);                 \
    } else if (is_integral(a.stype.dt)) {                                          \
      return ir_.make_value(spv::OpI##_Op, bool_type, a, b);                       \
    } else {                                                                       \
      QD_ASSERT(is_real(a.stype.dt));                                              \
      return ir_.make_value(spv::OpFOrd##_Op, bool_type, a, b);                    \
    }                                                                              \
  }

DEFINE_SPIRV_CMP_UOP(eq, Equal);
DEFINE_SPIRV_CMP_UOP(ne, NotEqual);

#define DEFINE_SPIRV_LOGICAL_OP(_OpName, _Op)                                                                   \
  Value SpirvOperations::_OpName(Value a, Value b) {                                                            \
    QD_ASSERT(a.stype.id == b.stype.id);                                                                        \
    if (a.stype.id == ir_.bool_type().id) {                                                                     \
      return ir_.make_value(spv::OpLogical##_Op, ir_.bool_type(), a, b);                                        \
    } else if (is_integral(a.stype.dt)) {                                                                       \
      Value val_a = ir_.make_value(spv::OpINotEqual, ir_.bool_type(), a, ir_.int_immediate_number(a.stype, 0)); \
      Value val_b = ir_.make_value(spv::OpINotEqual, ir_.bool_type(), b, ir_.int_immediate_number(b.stype, 0)); \
      Value val_ret = ir_.make_value(spv::OpLogical##_Op, ir_.bool_type(), val_a, val_b);                       \
      return cast(a.stype, val_ret);                                                                            \
    } else {                                                                                                    \
      QD_ERROR("Logical ops on real types are not supported.");                                                 \
      return Value();                                                                                           \
    }                                                                                                           \
  }

DEFINE_SPIRV_LOGICAL_OP(logical_and, And);
DEFINE_SPIRV_LOGICAL_OP(logical_or, Or);

Value SpirvOperations::bit_field_extract(Value base, Value offset, Value count) {
  QD_ASSERT(is_integral(base.stype.dt));
  QD_ASSERT(is_integral(offset.stype.dt));
  QD_ASSERT(is_integral(count.stype.dt));
  return ir_.make_value(spv::OpBitFieldUExtract, base.stype, base, offset, count);
}

Value SpirvOperations::select(Value cond, Value a, Value b) {
  QD_ASSERT(a.stype.id == b.stype.id);
  QD_ASSERT(cond.stype.id == ir_.bool_type().id);
  return ir_.make_value(spv::OpSelect, a.stype, cond, a, b);
}

Value SpirvOperations::cast(const SType &dst_type, Value value) {
  QD_ASSERT(value.stype.id > 0U);
  if (value.stype.id == dst_type.id)
    return value;
  const DataType &from = value.stype.dt;
  const DataType &to = dst_type.dt;
  if (from->is_primitive(PrimitiveTypeID::u1)) {  // Bool
    if (is_integral(to) && is_signed(to)) {       // Bool -> Int
      return select(value, ir_.int_immediate_number(dst_type, 1), ir_.int_immediate_number(dst_type, 0));
    } else if (is_integral(to) && is_unsigned(to)) {  // Bool -> UInt
      return select(value, ir_.uint_immediate_number(dst_type, 1), ir_.uint_immediate_number(dst_type, 0));
    } else if (is_real(to)) {  // Bool -> Float
      return ir_.make_value(
          spv::OpConvertUToF, dst_type,
          select(value, ir_.uint_immediate_number(ir_.u32_type(), 1), ir_.uint_immediate_number(ir_.u32_type(), 0)));
    } else {
      QD_ERROR("do not support type cast from {} to {}", from.to_string(), to.to_string());
      return Value();
    }
  } else if (to->is_primitive(PrimitiveTypeID::u1)) {  // Bool
    if (is_integral(from) && is_signed(from)) {        // Int -> Bool
      return ne(value, ir_.int_immediate_number(value.stype, 0));
    } else if (is_integral(from) && is_unsigned(from)) {  // UInt -> Bool
      return ne(value, ir_.uint_immediate_number(value.stype, 0));
    } else {
      QD_ERROR("do not support type cast from {} to {}", from.to_string(), to.to_string());
      return Value();
    }
  } else if (is_integral(from) && is_integral(to)) {
    auto ret = value;

    if (data_type_bits(from) == data_type_bits(to)) {
      // Same width conversion
      ret = ir_.make_value(spv::OpBitcast, dst_type, ret);
    } else {
      // Different width
      // Step 1. Sign extend / truncate value to width of `to`
      // Step 2. Bitcast to signess of `to`
      auto get_signed_type = [](DataType dt) -> DataType {
        // Create a output signed type with the same width as `dt`
        if (data_type_bits(dt) == 8)
          return PrimitiveType::i8;
        else if (data_type_bits(dt) == 16)
          return PrimitiveType::i16;
        else if (data_type_bits(dt) == 32)
          return PrimitiveType::i32;
        else if (data_type_bits(dt) == 64)
          return PrimitiveType::i64;
        else
          return PrimitiveType::unknown;
      };
      auto get_unsigned_type = [](DataType dt) -> DataType {
        // Create a output unsigned type with the same width as `dt`
        if (data_type_bits(dt) == 8)
          return PrimitiveType::u8;
        else if (data_type_bits(dt) == 16)
          return PrimitiveType::u16;
        else if (data_type_bits(dt) == 32)
          return PrimitiveType::u32;
        else if (data_type_bits(dt) == 64)
          return PrimitiveType::u64;
        else
          return PrimitiveType::unknown;
      };

      DataType intermediate_dt;
      if (is_signed(from)) {
        intermediate_dt = get_signed_type(to);
        ret = ir_.make_value(spv::OpSConvert, ir_.get_primitive_type(intermediate_dt), ret);
      } else {
        intermediate_dt = get_unsigned_type(to);
        ret = ir_.make_value(spv::OpUConvert, ir_.get_primitive_type(intermediate_dt), ret);
      }

      // OpBitcast(T, T) is invalid per SPIR-V spec ("Result Type must not equal Operand Type"). When the
      // intermediate dtype (same signedness as the source but width-matched to the destination) already
      // matches the caller's destination type, skip the trailing bitcast so the widening / narrowing SConvert
      // / UConvert above is the final instruction. The trailing bitcast still runs in the mixed-signedness
      // case, which is the scenario it was written for.
      if (intermediate_dt != to) {
        ret = ir_.make_value(spv::OpBitcast, dst_type, ret);
      }
    }

    return ret;
  } else if (is_real(from) && is_integral(to) && is_signed(to)) {  // Float -> Int
    return ir_.make_value(spv::OpConvertFToS, dst_type, value);
  } else if (is_real(from) && is_integral(to) && is_unsigned(to)) {  // Float -> UInt
    return ir_.make_value(spv::OpConvertFToU, dst_type, value);
  } else if (is_integral(from) && is_signed(from) && is_real(to)) {  // Int -> Float
    return ir_.make_value(spv::OpConvertSToF, dst_type, value);
  } else if (is_integral(from) && is_unsigned(from) && is_real(to)) {  // UInt -> Float
    return ir_.make_value(spv::OpConvertUToF, dst_type, value);
  } else if (is_real(from) && is_real(to)) {  // Float -> Float
    return ir_.make_value(spv::OpFConvert, dst_type, value);
  } else {
    QD_ERROR("do not support type cast from {} to {}", from.to_string(), to.to_string());
    return Value();
  }
}

void SpirvOperations::set_work_group_size(const std::array<int, 3> group_size) {
  Value size_x = ir_.uint_immediate_number(ir_.u32_type(), static_cast<uint64_t>(group_size[0]));
  Value size_y = ir_.uint_immediate_number(ir_.u32_type(), static_cast<uint64_t>(group_size[1]));
  Value size_z = ir_.uint_immediate_number(ir_.u32_type(), static_cast<uint64_t>(group_size[2]));

  if (gl_work_group_size_.id == 0) {
    gl_work_group_size_ = ir_.new_value(SType(), ValueKind::kNormal);
  }
  ir_.declare_global(spv::OpConstantComposite, ir_.v3_u32_type(), gl_work_group_size_, size_x, size_y, size_z);
  ir_.decorate(spv::OpDecorate, gl_work_group_size_, spv::DecorationBuiltIn, spv::BuiltInWorkgroupSize);
}

Value SpirvOperations::get_work_group_id(uint32_t dim_index) {
  QD_ASSERT(dim_index < 3);
  return ir_.call_glsl_u32_to_u32(get_work_group_id_fn_id_, "get_work_group_id", dim_index);
}

Value SpirvOperations::get_num_work_groups(uint32_t dim_index) {
  if (gl_num_work_groups_.id == 0) {
    SType ptr_type = ir_.get_pointer_type(ir_.v3_u32_type(), spv::StorageClassInput);
    gl_num_work_groups_ = ir_.new_value(ptr_type, ValueKind::kVectorPtr);
    ir_.register_entry_point_input(gl_num_work_groups_);
    ir_.declare_global(spv::OpVariable, ptr_type, gl_num_work_groups_, spv::StorageClassInput);
    ir_.decorate(spv::OpDecorate, gl_num_work_groups_, spv::DecorationBuiltIn, spv::BuiltInNumWorkgroups);
  }
  SType pint_type = ir_.get_pointer_type(ir_.u32_type(), spv::StorageClassInput);
  Value ptr = ir_.make_value(spv::OpAccessChain, pint_type, gl_num_work_groups_,
                             ir_.uint_immediate_number(ir_.u32_type(), static_cast<uint64_t>(dim_index)));

  return ir_.make_value(spv::OpLoad, ir_.u32_type(), ptr);
}

Value SpirvOperations::get_local_invocation_id(uint32_t dim_index) {
  if (gl_local_invocation_id_.id == 0) {
    SType ptr_type = ir_.get_pointer_type(ir_.v3_u32_type(), spv::StorageClassInput);
    gl_local_invocation_id_ = ir_.new_value(ptr_type, ValueKind::kVectorPtr);
    ir_.register_entry_point_input(gl_local_invocation_id_);
    ir_.declare_global(spv::OpVariable, ptr_type, gl_local_invocation_id_, spv::StorageClassInput);
    ir_.decorate(spv::OpDecorate, gl_local_invocation_id_, spv::DecorationBuiltIn, spv::BuiltInLocalInvocationId);
  }
  SType pint_type = ir_.get_pointer_type(ir_.u32_type(), spv::StorageClassInput);
  Value ptr = ir_.make_value(spv::OpAccessChain, pint_type, gl_local_invocation_id_,
                             ir_.uint_immediate_number(ir_.u32_type(), static_cast<uint64_t>(dim_index)));

  return ir_.make_value(spv::OpLoad, ir_.u32_type(), ptr);
}

Value SpirvOperations::get_global_invocation_id(uint32_t dim_index) {
  if (gl_global_invocation_id_.id == 0) {
    SType ptr_type = ir_.get_pointer_type(ir_.v3_u32_type(), spv::StorageClassInput);
    gl_global_invocation_id_ = ir_.new_value(ptr_type, ValueKind::kVectorPtr);
    ir_.register_entry_point_input(gl_global_invocation_id_);
    ir_.declare_global(spv::OpVariable, ptr_type, gl_global_invocation_id_, spv::StorageClassInput);
    ir_.decorate(spv::OpDecorate, gl_global_invocation_id_, spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);
  }
  SType pint_type = ir_.get_pointer_type(ir_.u32_type(), spv::StorageClassInput);
  Value ptr = ir_.make_value(spv::OpAccessChain, pint_type, gl_global_invocation_id_,
                             ir_.uint_immediate_number(ir_.u32_type(), static_cast<uint64_t>(dim_index)));

  return ir_.make_value(spv::OpLoad, ir_.u32_type(), ptr);
}

Value SpirvOperations::get_subgroup_invocation_id() {
  if (subgroup_local_invocation_id_.id == 0) {
    SType ptr_type = ir_.get_pointer_type(ir_.u32_type(), spv::StorageClassInput);
    subgroup_local_invocation_id_ = ir_.new_value(ptr_type, ValueKind::kVariablePtr);
    ir_.declare_global(spv::OpVariable, ptr_type, subgroup_local_invocation_id_, spv::StorageClassInput);
    ir_.decorate(spv::OpDecorate, subgroup_local_invocation_id_, spv::DecorationBuiltIn,
                 spv::BuiltInSubgroupLocalInvocationId);
    ir_.global_values.push_back(subgroup_local_invocation_id_);
  }

  return ir_.make_value(spv::OpLoad, ir_.u32_type(), subgroup_local_invocation_id_);
}

Value SpirvOperations::float_atomic(AtomicOpType op_type, Value addr_ptr, Value data, const DataType &dt) {
  // Use dt-derived type instead of ir_.f32_type() so FMin/FMax work for f16/f64.
  auto float_type = ir_.get_primitive_type(dt);
  if (op_type == AtomicOpType::add) {
    return atomic_operation(addr_ptr, data, [&](Value lhs, Value rhs) { return add(lhs, rhs); }, dt);
  } else if (op_type == AtomicOpType::sub) {
    return atomic_operation(addr_ptr, data, [&](Value lhs, Value rhs) { return sub(lhs, rhs); }, dt);
  } else if (op_type == AtomicOpType::mul) {
    return atomic_operation(addr_ptr, data, [&](Value lhs, Value rhs) { return mul(lhs, rhs); }, dt);
  } else if (op_type == AtomicOpType::min) {
    return atomic_operation(
        addr_ptr, data, [&](Value lhs, Value rhs) { return ir_.call_glsl450(float_type, /*FMin*/ 37, lhs, rhs); }, dt);
  } else if (op_type == AtomicOpType::max) {
    return atomic_operation(
        addr_ptr, data, [&](Value lhs, Value rhs) { return ir_.call_glsl450(float_type, /*FMax*/ 40, lhs, rhs); }, dt);
  } else {
    QD_NOT_IMPLEMENTED
  }
}

Value SpirvOperations::integer_atomic(AtomicOpType op_type, Value addr_ptr, Value data, const DataType &dt) {
  if (op_type == AtomicOpType::mul) {
    return atomic_operation(addr_ptr, data, [&](Value lhs, Value rhs) { return mul(lhs, rhs); }, dt);
  } else {
    QD_NOT_IMPLEMENTED
  }
}

Value SpirvOperations::atomic_operation(Value addr_ptr,
                                        Value data,
                                        std::function<Value(Value, Value)> op,
                                        const DataType &dt) {
  SType out_type = ir_.get_primitive_type(dt);
  // Device-buffer pointers are uint-typed (from at_buffer), so CAS uses uint.
  // Workgroup (shared) pointers keep their original type (e.g. i32). Using uint
  // on a signed pointer causes Metal's atomic_compare_exchange to reject the
  // shader due to signed/unsigned type mismatch.
  const bool is_workgroup = addr_ptr.stype.storage_class == spv::StorageClassWorkgroup;
  SType res_type = is_workgroup ? out_type : ir_.get_primitive_uint_type(dt);
  Value ret_val_int = ir_.alloca_variable(res_type);

  // do-while
  Label head = ir_.new_label();
  Label body = ir_.new_label();
  Label branch_true = ir_.new_label();
  Label branch_false = ir_.new_label();
  Label merge = ir_.new_label();
  Label exit = ir_.new_label();

  ir_.make_inst(spv::OpBranch, head);
  ir_.start_label(head);
  ir_.make_inst(spv::OpLoopMerge, branch_true, merge, 0);
  ir_.make_inst(spv::OpBranch, body);
  ir_.make_inst(spv::OpLabel, body);
  // while (true)
  {
    // Use OpAtomicLoad so SPIRV-Cross emits a function call expression
    // (atomic_load_explicit) that it cannot inline.  A plain OpLoad would
    // be inlined as a device-memory dereference, causing SPIRV-Cross's CAS
    // emulation loop to re-read (and see the post-CAS value), breaking the
    // compare-and-swap logic on Metal.
    Value old_val = ir_.make_value(spv::OpAtomicLoad, res_type, addr_ptr,
                                   /*scope=*/ir_.const_i32_one_,
                                   /*semantics=*/ir_.const_i32_zero_);
    // Bitcast uint<->float for the operation. Skip when types already match
    // (integer workgroup path where res_type == out_type).
    Value old_data_value = (out_type.id != res_type.id) ? ir_.make_value(spv::OpBitcast, out_type, old_val) : old_val;
    Value new_data_value = op(old_data_value, data);
    Value new_val =
        (out_type.id != res_type.id) ? ir_.make_value(spv::OpBitcast, res_type, new_data_value) : new_data_value;
    // int loaded = atomicCompSwap(vals[0], old, new);
    /*
    * Don't need this part, theoretically
    auto semantics = uint_imm ediate_number(
        ir_.u32_type(), spv::MemorySemanticsAcquireReleaseMask |
                       spv::MemorySemanticsUniformMemoryMask);
    ir_.make_inst(spv::OpMemoryBarrier, ir_.const_i32_one_, semantics);
    */
    Value loaded = ir_.make_value(spv::OpAtomicCompareExchange, res_type, addr_ptr,
                                  /*scope=*/ir_.const_i32_one_, /*semantics if equal=*/ir_.const_i32_zero_,
                                  /*semantics if unequal=*/ir_.const_i32_zero_, new_val, old_val);
    // bool ok = (loaded == old);
    Value ok = ir_.make_value(spv::OpIEqual, ir_.bool_type(), loaded, old_val);
    // int ret_val_int = loaded;
    ir_.store_variable(ret_val_int, loaded);
    // if (ok)
    ir_.make_inst(spv::OpSelectionMerge, branch_false, 0);
    ir_.make_inst(spv::OpBranchConditional, ok, branch_true, branch_false);
    {
      ir_.make_inst(spv::OpLabel, branch_true);
      ir_.make_inst(spv::OpBranch, exit);
    }
    // else
    {
      ir_.make_inst(spv::OpLabel, branch_false);
      ir_.make_inst(spv::OpBranch, merge);
    }
    // continue;
    ir_.make_inst(spv::OpLabel, merge);
    ir_.make_inst(spv::OpBranch, head);
  }
  ir_.start_label(exit);

  Value ret_loaded = ir_.load_variable(ret_val_int, res_type);
  return (out_type.id != res_type.id) ? ir_.make_value(spv::OpBitcast, out_type, ret_loaded) : ret_loaded;
}

Value SpirvOperations::rand_u32(Value global_tmp_) {
  if (!init_rand_) {
    init_random_function(global_tmp_);
  }

  Value _11u = ir_.uint_immediate_number(ir_.u32_type(), 11u);
  Value _19u = ir_.uint_immediate_number(ir_.u32_type(), 19u);
  Value _8u = ir_.uint_immediate_number(ir_.u32_type(), 8u);
  Value _1000000007u = ir_.uint_immediate_number(ir_.u32_type(), 1000000007u);
  Value tmp0 = ir_.load_variable(rand_x_, ir_.u32_type());
  Value tmp1 = ir_.make_value(spv::OpShiftLeftLogical, ir_.u32_type(), tmp0, _11u);
  Value tmp_t = ir_.make_value(spv::OpBitwiseXor, ir_.u32_type(), tmp0, tmp1);  // t
  ir_.store_variable(rand_x_, ir_.load_variable(rand_y_, ir_.u32_type()));
  ir_.store_variable(rand_y_, ir_.load_variable(rand_z_, ir_.u32_type()));
  Value tmp_w = ir_.load_variable(rand_w_, ir_.u32_type());  // reuse w
  ir_.store_variable(rand_z_, tmp_w);
  Value tmp2 = ir_.make_value(spv::OpShiftRightLogical, ir_.u32_type(), tmp_w, _19u);
  Value tmp3 = ir_.make_value(spv::OpBitwiseXor, ir_.u32_type(), tmp_w, tmp2);
  Value tmp4 = ir_.make_value(spv::OpShiftRightLogical, ir_.u32_type(), tmp_t, _8u);
  Value tmp5 = ir_.make_value(spv::OpBitwiseXor, ir_.u32_type(), tmp_t, tmp4);
  Value new_w = ir_.make_value(spv::OpBitwiseXor, ir_.u32_type(), tmp3, tmp5);
  ir_.store_variable(rand_w_, new_w);
  Value val = ir_.make_value(spv::OpIMul, ir_.u32_type(), new_w, _1000000007u);

  return val;
}

Value SpirvOperations::rand_f32(Value global_tmp_) {
  if (!init_rand_) {
    init_random_function(global_tmp_);
  }

  Value _1_4294967296f = ir_.float_immediate_number(ir_.f32_type(), 1.0f / 4294967296.0f);
  Value tmp0 = rand_u32(global_tmp_);
  Value tmp1 = cast(ir_.f32_type(), tmp0);
  Value val = mul(tmp1, _1_4294967296f);

  return val;
}

Value SpirvOperations::rand_i32(Value global_tmp_) {
  if (!init_rand_) {
    init_random_function(global_tmp_);
  }

  Value tmp0 = rand_u32(global_tmp_);
  Value val = cast(ir_.i32_type(), tmp0);
  return val;
}

void SpirvOperations::init_random_function(Value global_tmp_) {
  // variables declare
  SType local_type = ir_.get_pointer_type(ir_.u32_type(), spv::StorageClassPrivate);
  rand_x_ = ir_.new_value(local_type, ValueKind::kVariablePtr);
  rand_y_ = ir_.new_value(local_type, ValueKind::kVariablePtr);
  rand_z_ = ir_.new_value(local_type, ValueKind::kVariablePtr);
  rand_w_ = ir_.new_value(local_type, ValueKind::kVariablePtr);
  ir_.global_values.push_back(rand_x_);
  ir_.global_values.push_back(rand_y_);
  ir_.global_values.push_back(rand_z_);
  ir_.global_values.push_back(rand_w_);
  ir_.declare_global(spv::OpVariable, local_type, rand_x_, spv::StorageClassPrivate);
  ir_.declare_global(spv::OpVariable, local_type, rand_y_, spv::StorageClassPrivate);
  ir_.declare_global(spv::OpVariable, local_type, rand_z_, spv::StorageClassPrivate);
  ir_.declare_global(spv::OpVariable, local_type, rand_w_, spv::StorageClassPrivate);
  ir_.debug_name(spv::OpName, rand_x_, "_rand_x");
  ir_.debug_name(spv::OpName, rand_y_, "_rand_y");
  ir_.debug_name(spv::OpName, rand_z_, "_rand_z");
  ir_.debug_name(spv::OpName, rand_w_, "_rand_w");
  SType gtmp_type = ir_.get_pointer_type(ir_.u32_type(), spv::StorageClassStorageBuffer);
  Value rand_gtmp_ = ir_.new_value(gtmp_type, ValueKind::kVariablePtr);
  ir_.debug_name(spv::OpName, rand_gtmp_, "rand_gtmp");

  auto load_var = [&](Value pointer, const SType &res_type) {
    QD_ASSERT(pointer.flag == ValueKind::kVariablePtr || pointer.flag == ValueKind::kStructArrayPtr);
    Value ret = ir_.new_value(res_type, ValueKind::kNormal);
    ir_.make_function_header_inst(spv::OpLoad, res_type, ret, pointer);
    return ret;
  };

  auto store_var = [&](Value pointer, Value value) {
    QD_ASSERT(pointer.flag == ValueKind::kVariablePtr);
    QD_ASSERT(value.stype.id == pointer.stype.element_type_id);
    ir_.make_function_header_inst(spv::OpStore, pointer, value);
  };

  // Constant Number
  Value _7654321u = ir_.uint_immediate_number(ir_.u32_type(), 7654321u);
  Value _1234567u = ir_.uint_immediate_number(ir_.u32_type(), 1234567u);
  Value _9723451u = ir_.uint_immediate_number(ir_.u32_type(), 9723451u);
  Value _123456789u = ir_.uint_immediate_number(ir_.u32_type(), 123456789u);
  Value _1000000007u = ir_.uint_immediate_number(ir_.u32_type(), 1000000007u);
  Value _362436069u = ir_.uint_immediate_number(ir_.u32_type(), 362436069u);
  Value _521288629u = ir_.uint_immediate_number(ir_.u32_type(), 521288629u);
  Value _88675123u = ir_.uint_immediate_number(ir_.u32_type(), 88675123u);
  Value _1 = ir_.int_immediate_number(ir_.u32_type(), 1);
  Value _1024 = ir_.int_immediate_number(ir_.u32_type(), 1024);

  // init_rand_ segment (inline to main)
  // ad-hoc: hope no kernel will use more than 1024 gtmp variables...
  ir_.make_function_header_inst(spv::OpAccessChain, gtmp_type, rand_gtmp_, global_tmp_, ir_.const_i32_zero_, _1024);
  // Get gl_GlobalInvocationID.x, assert it has be visited
  // (in generate_serial_kernel/generate_range_for_kernel
  SType pint_type = ir_.get_pointer_type(ir_.u32_type(), spv::StorageClassInput);
  Value tmp0 = ir_.new_value(pint_type, ValueKind::kVariablePtr);
  ir_.make_function_header_inst(spv::OpAccessChain, pint_type, tmp0, gl_global_invocation_id_,
                                ir_.uint_immediate_number(ir_.u32_type(), 0));
  Value tmp1 = load_var(tmp0, ir_.u32_type());
  Value tmp2_ = load_var(rand_gtmp_, ir_.u32_type());
  Value tmp2 = ir_.new_value(ir_.u32_type(), ValueKind::kNormal);
  ir_.make_function_header_inst(spv::OpBitcast, ir_.u32_type(), tmp2, tmp2_);
  Value tmp3 = ir_.new_value(ir_.u32_type(), ValueKind::kNormal);
  ir_.make_function_header_inst(spv::OpIAdd, ir_.u32_type(), tmp3, _7654321u, tmp1);
  Value tmp4 = ir_.new_value(ir_.u32_type(), ValueKind::kNormal);
  ir_.make_function_header_inst(spv::OpIMul, ir_.u32_type(), tmp4, _9723451u, tmp2);
  Value tmp5 = ir_.new_value(ir_.u32_type(), ValueKind::kNormal);
  ir_.make_function_header_inst(spv::OpIAdd, ir_.u32_type(), tmp5, _1234567u, tmp4);
  Value tmp6 = ir_.new_value(ir_.u32_type(), ValueKind::kNormal);
  ir_.make_function_header_inst(spv::OpIMul, ir_.u32_type(), tmp6, tmp3, tmp5);
  Value tmp7 = ir_.new_value(ir_.u32_type(), ValueKind::kNormal);
  ir_.make_function_header_inst(spv::OpIMul, ir_.u32_type(), tmp7, _123456789u, tmp6);
  Value tmp8 = ir_.new_value(ir_.u32_type(), ValueKind::kNormal);
  ir_.make_function_header_inst(spv::OpIMul, ir_.u32_type(), tmp8, _1000000007u, tmp7);
  store_var(rand_x_, tmp8);
  store_var(rand_y_, _362436069u);
  store_var(rand_z_, _521288629u);
  store_var(rand_w_, _88675123u);

  // enum spv::Op add_op = spv::OpIAdd;
  bool use_atomic_increment = false;

  if (use_atomic_increment) {
    Value tmp9 = ir_.new_value(ir_.u32_type(), ValueKind::kNormal);
    ir_.make_function_header_inst(spv::Op::OpAtomicIIncrement, ir_.u32_type(), tmp9, rand_gtmp_,
                                  /*scope_id*/ ir_.const_i32_one_,
                                  /*semantics*/ ir_.const_i32_zero_);
  } else {
    // Yes, this is not an atomic operation, but just fine since no matter
    // how RAND_STATE changes, `gl_GlobalInvocationID.x` can still help
    // us to set different seeds for different threads.
    // Discussion:
    // https://github.com/taichi-dev/taichi/pull/912#discussion_r419021918
    Value tmp9 = load_var(rand_gtmp_, ir_.u32_type());
    Value tmp10 = ir_.new_value(ir_.u32_type(), ValueKind::kNormal);
    ir_.make_function_header_inst(spv::Op::OpIAdd, ir_.u32_type(), tmp10, tmp9, _1);
    store_var(rand_gtmp_, tmp10);
  }

  init_rand_ = true;
}

void SpirvOperations::call_debugprintf(std::string formats, const std::vector<Value> &args) {
  // Import lazily: an unused debugPrintf import can cause Metal compilation to fail.
  if (!debug_printf_.id) {
    debug_printf_ = ir_.ext_inst_import("NonSemantic.DebugPrintf");
  }
  Value format_str = ir_.debug_string(formats);
  Value val = ir_.new_value(ir_.void_type(), ValueKind::kNormal);
  std::vector<uint32_t> argument_ids;
  for (const auto &arg : args) {
    argument_ids.push_back(arg.id);
  }
  ir_.make_inst(spv::OpExtInst, ir_.void_type(), val, debug_printf_, 1, format_str, argument_ids);
}

}  // namespace quadrants::lang::spirv
