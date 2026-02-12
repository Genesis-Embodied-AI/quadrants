#include <quadrants/program/launch_context_builder.h>
#define TI_RUNTIME_HOST
#include <quadrants/program/context.h>
#undef TI_RUNTIME_HOST
#include "fp16.h"

namespace quadrants::lang {

namespace {
template <typename T>
inline std::vector<T> concatenate_vector(const std::vector<T> &lhs,
                                         const std::vector<T> &rhs) {
  std::vector<T> result;
  result.assign(lhs.begin(), lhs.end());
  result.insert(result.end(), rhs.begin(), rhs.end());
  return result;
}
}  // namespace

LaunchContextBuilder::LaunchContextBuilder(CallableBase *kernel)
    : kernel_(kernel),
      owned_ctx_(std::make_unique<RuntimeContext>()),
      ctx_(owned_ctx_.get()),
      arg_buffer_(std::make_unique<char[]>(kernel->args_size)),
      result_buffer_(std::make_unique<char[]>(kernel->ret_size)),
      ret_type_(kernel->ret_type),
      arg_buffer_size(kernel->args_size),
      args_type(kernel->args_type),
      result_buffer_size(kernel->ret_size) {
  ctx_->result_buffer = (uint64 *)result_buffer_.get();
  ctx_->arg_buffer = arg_buffer_.get();
}

void LaunchContextBuilder::copy(const LaunchContextBuilder &other) {
  TI_ASSERT(kernel_ == other.kernel_);
  TI_ASSERT(array_runtime_sizes.empty());
  TI_ASSERT(device_allocation_type.empty());
  TI_ASSERT(array_ptrs.empty());
  std::memcpy(arg_buffer_.get(), other.arg_buffer_.get(), kernel_->args_size);
  array_runtime_sizes = other.array_runtime_sizes;
  device_allocation_type = other.device_allocation_type;
  array_ptrs = other.array_ptrs;
}

void LaunchContextBuilder::set_arg_float(int arg_id, float64 d) {
  auto dt = kernel_->args_type->get_element_type(std::array{arg_id});
  TI_ASSERT_INFO(dt->is<PrimitiveType>(),
                 "Assigning scalar value to external (numpy) array argument is "
                 "not allowed.");

  PrimitiveTypeID typeId = dt->as<PrimitiveType>()->type;

  switch (typeId) {
#define PER_C_TYPE(tp, ctype)  \
  case PrimitiveTypeID::tp:    \
    set_arg(arg_id, (ctype)d); \
    break;
#include "quadrants/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
    case PrimitiveTypeID::f16: {
      uint16 half = fp16_ieee_from_fp32_value((float32)d);
      set_arg(arg_id, half);
      break;
    }
    default:
      TI_NOT_IMPLEMENTED
  }
}

void LaunchContextBuilder::set_args_float(const std::vector<int> &args_id,
                                          const std::vector<float64> &vec) {
  const size_t num_arrs = args_id.size();
  TI_ASSERT(num_arrs == vec.size());
  for (int i = 0; i < num_arrs; i++) {
    set_arg_float(args_id[i], vec[i]);
  }
}

template <typename T, std::size_t N>
void LaunchContextBuilder::set_struct_arg(const std::array<int, N> &arg_indices,
                                          T d) {
  auto dt = kernel_->args_type->get_element_type(arg_indices);

  TI_ASSERT(dt->template is<PrimitiveType>() || dt->template is<PointerType>());
  if (dt->template is<PointerType>()) {
    set_struct_arg_impl(arg_indices, (uint64)d);
    return;
  }
  PrimitiveTypeID typeId = dt->template as<PrimitiveType>()->type;

  switch (typeId) {
#define PER_C_TYPE(tp, ctype)                   \
  case PrimitiveTypeID::tp:                     \
    set_struct_arg_impl(arg_indices, (ctype)d); \
    break;
#include "quadrants/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
    case PrimitiveTypeID::f16: {
      uint16 half = fp16_ieee_from_fp32_value((float32)d);
      set_struct_arg_impl(arg_indices, half);
      break;
    }
    default:
      TI_NOT_IMPLEMENTED
  }
}

template <std::size_t... I>
constexpr auto vec_to_array(const std::vector<int> &v,
                            std::index_sequence<I...>) {
  return std::array<int, sizeof...(I)>{v[I]...};
}

template <typename T>
void LaunchContextBuilder::set_struct_arg(const std::vector<int> &arg_indices,
                                          T v) {
  switch (arg_indices.size()) {
#define CASE(N)                                                       \
  case N: {                                                           \
    return set_struct_arg(                                            \
        vec_to_array(arg_indices, std::make_index_sequence<N>{}), v); \
  }
    CASE(1)
    CASE(2)
    CASE(3)
    CASE(4)
    CASE(5)
    CASE(6)
    CASE(7)
#undef CASE
    default:
      TI_NOT_IMPLEMENTED
  }
}

void LaunchContextBuilder::set_ndarray_ptrs(int arg_id,
                                            uint64 data_ptr,
                                            uint64 grad_ptr) {
  set_struct_arg(std::array{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY},
                 data_ptr);
  if (kernel_->nested_parameters[arg_id].needs_grad) {
    set_struct_arg(std::array{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY},
                   grad_ptr);
  }
}

template void LaunchContextBuilder::set_struct_arg(const std::array<int, 1> &,
                                                   uint64);
template void LaunchContextBuilder::set_struct_arg(const std::array<int, 1> &,
                                                   int64);
template void LaunchContextBuilder::set_struct_arg(const std::array<int, 1> &,
                                                   float64);
template void LaunchContextBuilder::set_struct_arg(const std::array<int, 2> &,
                                                   uint64);
template void LaunchContextBuilder::set_struct_arg(const std::array<int, 2> &,
                                                   int64);
template void LaunchContextBuilder::set_struct_arg(const std::array<int, 2> &,
                                                   float64);
template void LaunchContextBuilder::set_struct_arg(const std::array<int, 3> &,
                                                   uint64);
template void LaunchContextBuilder::set_struct_arg(const std::array<int, 3> &,
                                                   int64);
template void LaunchContextBuilder::set_struct_arg(const std::array<int, 3> &,
                                                   float64);
template void LaunchContextBuilder::set_struct_arg(const std::vector<int> &,
                                                   uint64);
template void LaunchContextBuilder::set_struct_arg(const std::vector<int> &,
                                                   int64);
template void LaunchContextBuilder::set_struct_arg(const std::vector<int> &,
                                                   float64);

void LaunchContextBuilder::set_arg_int(int arg_id, int64 d) {
  auto dt = kernel_->args_type->get_element_type(std::array{arg_id});

  TI_ASSERT_INFO(dt->is<PrimitiveType>(),
                 "Assigning scalar value to external (numpy) array argument is "
                 "not allowed.");

  if (dt->is_primitive(PrimitiveTypeID::i32)) {
    set_arg(arg_id, (int32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    set_arg(arg_id, (int64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    set_arg(arg_id, (int8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    set_arg(arg_id, (int16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u1)) {
    set_arg(arg_id, (uint1)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    set_arg(arg_id, (uint8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    set_arg(arg_id, (uint16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    set_arg(arg_id, (uint32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    set_arg(arg_id, (uint64)d);
  } else {
    TI_INFO(dt->to_string());
    TI_NOT_IMPLEMENTED
  }
}

void LaunchContextBuilder::set_args_int(const std::vector<int> &args_id,
                                        const std::vector<int64> &vec) {
  const size_t num_arrs = args_id.size();
  TI_ASSERT(num_arrs == vec.size());
  for (int i = 0; i < num_arrs; i++) {
    set_arg_int(args_id[i], vec[i]);
  }
}

void LaunchContextBuilder::set_arg_uint(int arg_id, uint64 d) {
  set_arg_int(arg_id, d);
}

void LaunchContextBuilder::set_args_uint(const std::vector<int> &args_id,
                                         const std::vector<uint64> &vec) {
  const size_t num_arrs = args_id.size();
  TI_ASSERT(num_arrs == vec.size());
  for (int i = 0; i < num_arrs; i++) {
    set_arg_uint(args_id[i], vec[i]);
  }
}

template <>
void LaunchContextBuilder::set_arg<TypedConstant>(int arg_id, TypedConstant d) {
  if (is_real(d.dt)) {
    set_arg_float(arg_id, d.val_float());
  } else if (is_signed(d.dt)) {
    set_arg_int(arg_id, d.val_int());
  } else {
    set_arg_uint(arg_id, d.val_uint());
  }
}

template <typename T, std::size_t N>
void LaunchContextBuilder::set_struct_arg_impl(
    const std::array<int, N> &arg_indices,
    T v) {
  int offset = args_type->get_element_offset(arg_indices);
  TI_ASSERT(offset + sizeof(T) <= arg_buffer_size);
  *(T *)(ctx_->arg_buffer + offset) = v;
}

template <typename T>
T LaunchContextBuilder::get_arg(const std::vector<int> &i) {
  return get_struct_arg<T>(i);
}

template <typename T>
T LaunchContextBuilder::get_struct_arg(std::vector<int> arg_indices) {
  int offset = args_type->get_element_offset(arg_indices);
  TI_ASSERT(offset + sizeof(T) <= arg_buffer_size);
  return *(T *)(ctx_->arg_buffer + offset);
}

template <typename T>
void LaunchContextBuilder::set_arg(int arg_id, T v) {
  set_struct_arg_impl(std::array{arg_id}, v);
  set_array_device_allocation_type(arg_id, DevAllocType::kNone);
}

template <typename T>
T LaunchContextBuilder::get_ret(int i) {
  return quadrants_union_cast_with_different_sizes<T>(ctx_->result_buffer[i]);
}

#define PER_C_TYPE(type, ctype)                                            \
  template void LaunchContextBuilder::set_struct_arg_impl(                 \
      const std::array<int, 1> &arg_indices, ctype v);                     \
  template void LaunchContextBuilder::set_struct_arg_impl(                 \
      const std::array<int, 2> &arg_indices, ctype v);                     \
  template void LaunchContextBuilder::set_struct_arg_impl(                 \
      const std::array<int, 3> &arg_indices, ctype v);                     \
  template ctype LaunchContextBuilder::get_arg(const std::vector<int> &i); \
  template ctype LaunchContextBuilder::get_struct_arg(                     \
      std::vector<int> arg_indices);                                       \
  template void LaunchContextBuilder::set_arg(int i, ctype v);             \
  template ctype LaunchContextBuilder::get_ret(int i);
#include "quadrants/inc/data_type_with_c_type.inc.h"
PER_C_TYPE(gen, void *)  // Register void* as a valid type
#undef PER_C_TYPE

void LaunchContextBuilder::set_array_runtime_size(int i, uint64 size) {
  array_runtime_sizes[i] = size;
}

void LaunchContextBuilder::set_array_device_allocation_type(
    int i,
    DevAllocType usage) {
  device_allocation_type[i] = usage;
}

void LaunchContextBuilder::set_arg_external_array_with_shape(
    int arg_id,
    uintptr_t ptr,
    uint64 size,
    const std::vector<int64> &shape,
    uintptr_t grad_ptr) {
  TI_ASSERT_INFO(
      kernel_->nested_parameters[arg_id].is_array,
      "Assigning external (numpy) array to scalar argument is not allowed.");

  TI_ASSERT_INFO(shape.size() <= quadrants_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  array_ptrs[{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY}] = (void *)ptr;
  if (grad_ptr != 0) {
    array_ptrs[{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY}] =
        (void *)grad_ptr;
  }
  set_array_runtime_size(arg_id, size);
  set_array_device_allocation_type(arg_id, DevAllocType::kNone);
  for (int i = 0; i < shape.size(); i++) {
    set_struct_arg(std::array{arg_id, 0, i}, (int32)shape[i]);
  }
}

void LaunchContextBuilder::set_arg_ndarray(int arg_id, const Ndarray &arr) {
  intptr_t ptr = arr.get_device_allocation_ptr_as_int();
  TI_ASSERT_INFO(arr.shape.size() <= quadrants_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  set_arg_ndarray_impl(arg_id, ptr, arr.shape);
}

void LaunchContextBuilder::set_args_ndarray(
    const std::vector<int> &args_id,
    const std::vector<Ndarray *> &arrs) {
  const size_t num_arrs = args_id.size();

  array_ptrs.reserve(array_ptrs.size() + num_arrs);
  array_runtime_sizes.reserve(array_runtime_sizes.size() + num_arrs);
  device_allocation_type.reserve(device_allocation_type.size() + num_arrs);

  TI_ASSERT(num_arrs == args_id.size());
  for (int i = 0; i < num_arrs; i++) {
    const int arg_id = args_id[i];
    const Ndarray &arr = *arrs[i];

    intptr_t ptr = arr.get_device_allocation_ptr_as_int();
    array_ptrs[{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY}] = (void *)ptr;
    set_array_device_allocation_type(arg_id, DevAllocType::kNdarray);
    const std::vector<int> &shape = arr.shape;
    size_t total_size = 1;
    for (int32 i = 0; i < shape.size(); i++) {
      set_struct_arg(std::array{arg_id, 0, i}, (int32)shape[i]);
      total_size *= shape[i];
    }
    set_array_runtime_size(arg_id, total_size);
  }
}

void LaunchContextBuilder::set_arg_ndarray_with_grad(int arg_id,
                                                     const Ndarray &arr,
                                                     const Ndarray &arr_grad) {
  intptr_t ptr = arr.get_device_allocation_ptr_as_int();
  intptr_t ptr_grad = arr_grad.get_device_allocation_ptr_as_int();
  TI_ASSERT_INFO(arr.shape.size() <= quadrants_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  set_arg_ndarray_impl(arg_id, ptr, arr.shape, ptr_grad);
}

void LaunchContextBuilder::set_args_ndarray_with_grad(
    const std::vector<int> &args_id,
    const std::vector<Ndarray *> &arrs,
    const std::vector<Ndarray *> &arrs_grad) {
  const size_t num_arrs = args_id.size();
  TI_ASSERT(num_arrs == arrs.size());
  TI_ASSERT(num_arrs == arrs_grad.size());
  for (int i = 0; i < num_arrs; i++) {
    set_arg_ndarray_with_grad(args_id[i], *arrs[i], *arrs_grad[i]);
  }
}

void LaunchContextBuilder::set_arg_ndarray_impl(int arg_id,
                                                intptr_t devalloc_ptr,
                                                const std::vector<int> &shape,
                                                intptr_t devalloc_ptr_grad) {
  // Set array ptr
  array_ptrs[{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY}] =
      (void *)devalloc_ptr;
  if (devalloc_ptr_grad != 0) {
    array_ptrs[{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY}] =
        (void *)devalloc_ptr_grad;
  }
  // Set device allocation type and runtime size
  set_array_device_allocation_type(arg_id, DevAllocType::kNdarray);
  TI_ASSERT(shape.size() <= quadrants_max_num_indices);
  size_t total_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    set_struct_arg(std::array{arg_id, 0, i}, (int32)shape[i]);
    total_size *= shape[i];
  }
  set_array_runtime_size(arg_id, total_size);
}

void LaunchContextBuilder::set_arg_matrix(int arg_id, const Matrix &matrix) {
  int type_size = data_type_size(matrix.dtype());
  for (int i = 0; i < matrix.length(); i++) {
    switch (type_size) {
      case 1:
        set_struct_arg_impl(std::array{arg_id, i},
                            quadrants_union_cast_with_different_sizes<int8>(
                                reinterpret_cast<uint8_t *>(matrix.data())[i]));
        break;
      case 2:
        set_struct_arg_impl(
            std::array{arg_id, i},
            quadrants_union_cast_with_different_sizes<int16>(
                reinterpret_cast<uint16_t *>(matrix.data())[i]));
        break;
      case 4:
        set_struct_arg_impl(
            std::array{arg_id, i},
            quadrants_union_cast_with_different_sizes<int32>(
                reinterpret_cast<uint32_t *>(matrix.data())[i]));
        break;
      case 8:
        set_struct_arg_impl(
            std::array{arg_id, i},
            quadrants_union_cast_with_different_sizes<int64>(
                reinterpret_cast<uint64_t *>(matrix.data())[i]));
        break;
      default:
        TI_ERROR("Unsupported type size {}", type_size);
    }
  }
}

TypedConstant LaunchContextBuilder::fetch_ret(const std::vector<int> &index) {
  const Type *dt = ret_type_->get_element_type(index);
  int offset = ret_type_->get_element_offset(index);
  return fetch_ret_impl(offset, dt);
}

float64 LaunchContextBuilder::get_struct_ret_float(
    const std::vector<int> &index) {
  return fetch_ret(index).val_float();
}

int64 LaunchContextBuilder::get_struct_ret_int(const std::vector<int> &index) {
  return fetch_ret(index).val_int();
}

uint64 LaunchContextBuilder::get_struct_ret_uint(
    const std::vector<int> &index) {
  return fetch_ret(index).val_uint();
}

TypedConstant LaunchContextBuilder::fetch_ret_impl(int offset, const Type *dt) {
  TI_ASSERT(dt->is<PrimitiveType>());
  auto primitive_type = dt->as<PrimitiveType>();
  char *ptr = result_buffer_.get() + offset;
  switch (primitive_type->type) {
#define PER_C_TYPE(type, ctype) \
  case PrimitiveTypeID::type:   \
    return TypedConstant(*(ctype *)ptr);
#include "quadrants/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
    case PrimitiveTypeID::f16: {
      // first fetch the data as u16, and then convert it to f32
      uint16 half = *(uint16 *)ptr;
      return TypedConstant(fp16_ieee_to_fp32_value(half));
    }
    default:
      TI_NOT_IMPLEMENTED
  }
}

RuntimeContext &LaunchContextBuilder::get_context() {
  return *ctx_;
}

}  // namespace quadrants::lang
