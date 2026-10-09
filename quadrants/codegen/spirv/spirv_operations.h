#pragma once

#include "quadrants/codegen/spirv/spirv_ir_builder.h"

namespace quadrants::lang::spirv {

// Concrete shader computations and GPU operations implemented using the generic SPIR-V builder.
// Keep one instance per builder so repeated operations reuse their declarations and state.
class SpirvOperations {
 public:
  explicit SpirvOperations(IRBuilder &ir) : ir_(ir) {
  }

  // Expressions
  Value add(Value a, Value b);
  Value sub(Value a, Value b);
  Value mul(Value a, Value b);
  Value div(Value a, Value b);
  Value mod(Value a, Value b);
  Value eq(Value a, Value b);
  Value ne(Value a, Value b);
  Value lt(Value a, Value b);
  Value le(Value a, Value b);
  Value gt(Value a, Value b);
  Value ge(Value a, Value b);
  Value logical_and(Value a, Value b);
  Value logical_or(Value a, Value b);
  Value bit_field_extract(Value base, Value offset, Value count);
  Value select(Value cond, Value a, Value b);
  Value popcnt(Value x);

  // Create a cast that cast value to dst_type
  Value cast(const SType &dst_type, Value value);

  void set_work_group_size(const std::array<int, 3> group_size);
  Value get_num_work_groups(uint32_t dim_index);
  Value get_work_group_id(uint32_t dim_index);
  Value get_local_invocation_id(uint32_t dim_index);
  Value get_global_invocation_id(uint32_t dim_index);
  Value get_subgroup_invocation_id();

  Value float_atomic(AtomicOpType op_type, Value addr_ptr, Value data, const DataType &dt);
  Value integer_atomic(AtomicOpType op_type, Value addr_ptr, Value data, const DataType &dt);
  Value atomic_operation(Value addr_ptr, Value data, std::function<Value(Value, Value)> op, const DataType &dt);

  Value rand_u32(Value global_tmp);
  Value rand_f32(Value global_tmp);
  Value rand_i32(Value global_tmp);
  void call_debugprintf(std::string formats, const std::vector<Value> &args);

 private:
  void init_random_function(Value global_tmp);

  IRBuilder &ir_;
  Value gl_global_invocation_id_;
  Value gl_local_invocation_id_;
  Value gl_num_work_groups_;
  Value gl_work_group_size_;
  Value subgroup_local_invocation_id_;
  Value debug_printf_;

  // Cached ID and type information for the imported GLSL function.
  Value get_work_group_id_fn_id_;

  bool init_rand_{false};
  Value rand_x_;
  Value rand_y_;
  Value rand_z_;
  Value rand_w_;
};

}  // namespace quadrants::lang::spirv
