// This file will only be compiled into llvm bitcode by clang.
// The generated bitcode will likely get inlined for performance.

#if !defined(QD_INCLUDED) || !defined(_WIN32)
// The latest MSVC(Visual Studio 2019 version 16.10.1, MSVC 14.29.30037)
// uses llvm-11 as requirements. Check this link for details:
// https://github.com/microsoft/STL/blob/1866b848f0175c3361a916680a4318e7f0cc5482/stl/inc/yvals_core.h#L550-L561.
// However, we use llvm-10 for now and building will fail due to clang version
// mismatch. Therefore, we workaround this problem by define such flag to skip
// the version check.
// NOTE(#2428)
#if defined(_WIN32) || defined(_WIN64)
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <cstring>

#include "quadrants/inc/constants.h"
#include "quadrants/inc/cuda_kernel_utils.inc.h"
#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/ir/static_adstack_bound_reducer_device.h"
#include "quadrants/math/arithmetic.h"

struct RuntimeContext;
using assert_failed_type = void (*)(const char *);
using host_printf_type = void (*)(const char *, ...);
// In llvm 15, host_printf_type will be saved as ptr instead of ptr of
// FunctionType.
// Add dummy function to save function type for host_printf_type.
extern "C" void get_func_type_host_printf(const char *, ...) {
}

using host_vsnprintf_type = int (*)(char *, std::size_t, const char *, std::va_list);
using host_allocator_type = void *(*)(void *, std::size_t, std::size_t);
using RangeForTaskFunc = void(RuntimeContext *, const char *tls, int i);
using MeshForTaskFunc = void(RuntimeContext *, const char *tls, uint32_t i);
using parallel_for_type = void (*)(void *thread_pool,
                                   int splits,
                                   int num_desired_threads,
                                   void *context,
                                   void (*func)(void *, int thread_id, int i));

#if defined(__linux__) && !ARCH_cuda && defined(QD_ARCH_x64)
__asm__(".symver logf,logf@GLIBC_2.2.5");
__asm__(".symver powf,powf@GLIBC_2.2.5");
__asm__(".symver expf,expf@GLIBC_2.2.5");
#endif

// For accessing struct fields
#define STRUCT_FIELD(S, F)                              \
  extern "C" decltype(S::F) S##_get_##F(S *s) {         \
    return s->F;                                        \
  }                                                     \
  extern "C" decltype(S::F) *S##_get_ptr_##F(S *s) {    \
    return &(s->F);                                     \
  }                                                     \
  extern "C" void S##_set_##F(S *s, decltype(S::F) f) { \
    s->F = f;                                           \
  }

#define STRUCT_FIELD_ARRAY(S, F)                                                          \
  extern "C" std::remove_all_extents_t<decltype(S::F)> S##_get_##F(S *s, int i) {         \
    return s->F[i];                                                                       \
  }                                                                                       \
  extern "C" void S##_set_##F(S *s, int i, std::remove_all_extents_t<decltype(S::F)> f) { \
    s->F[i] = f;                                                                          \
  };

// For fetching struct fields from device to host
#define RUNTIME_STRUCT_FIELD(S, F)                                       \
  extern "C" void runtime_##S##_get_##F(LLVMRuntime *runtime, S *s) {    \
    runtime->set_result(quadrants_result_buffer_runtime_query_id, s->F); \
  }

#define RUNTIME_STRUCT_FIELD_ARRAY(S, F)                                     \
  extern "C" void runtime_##S##_get_##F(LLVMRuntime *runtime, S *s, int i) { \
    runtime->set_result(quadrants_result_buffer_runtime_query_id, s->F[i]);  \
  }

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint1 = bool;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using float32 = float;
using float64 = double;

using i8 = int8;
using i16 = int16;
using i32 = int32;
using i64 = int64;
using u1 = uint1;
using u8 = uint8;
using u16 = uint16;
using u32 = uint32;
using u64 = uint64;
using f32 = float32;
using f64 = float64;

using uint8 = uint8_t;
using Ptr = uint8 *;

using RuntimeContextArgType = long long;

#if ARCH_cuda || ARCH_amdgpu
extern "C" {

void __assertfail(const char *message, const char *file, i32 line, const char *function, std::size_t charSize);
};
#endif

template <typename T>
void locked_task(void *lock, const T &func);

template <typename T, typename G>
void locked_task(void *lock, const T &func, const G &test);

struct LLVMRuntime;
template <typename... Args>
void quadrants_printf(LLVMRuntime *runtime, const char *format, Args &&...args);

extern "C" {

// This is not really a runtime function. Include this in a function body to
// mark it as force no inline. Helpful when preventing inlining huge function
// bodies.
void mark_force_no_inline() {
}

i64 cuda_clock_i64() {
  return 0;
}

i64 amdgpu_clock_i64() {
  return 0;
}

i64 cpu_clock_i64() {
  return 0;
}

void system_mem_fence() {
}

#if ARCH_cuda
void cuda_vprintf(Ptr format, Ptr arg);
#endif

// Note that strlen is undefined on the CUDA backend, so we manually
// implement it here.
std::size_t quadrants_strlen(const char *str) {
  std::size_t len = 0;
  for (auto p = str; *p; p++)
    len++;
  return len;
}

#define DEFINE_UNARY_REAL_FUNC(F) \
  f32 F##_f32(f32 x) {            \
    return std::F(x);             \
  }                               \
  f64 F##_f64(f64 x) {            \
    return std::F(x);             \
  }

DEFINE_UNARY_REAL_FUNC(exp)
DEFINE_UNARY_REAL_FUNC(log)
DEFINE_UNARY_REAL_FUNC(tan)
DEFINE_UNARY_REAL_FUNC(tanh)
DEFINE_UNARY_REAL_FUNC(abs)
DEFINE_UNARY_REAL_FUNC(acos)
DEFINE_UNARY_REAL_FUNC(asin)
DEFINE_UNARY_REAL_FUNC(cos)
DEFINE_UNARY_REAL_FUNC(sin)

i32 abs_i32(i32 a) {
  return a >= 0 ? a : -a;
}

i64 abs_i64(i64 a) {
  return a >= 0 ? a : -a;
}

u16 min_u16(u16 a, u16 b) {
  return a < b ? a : b;
}

i16 min_i16(i16 a, i16 b) {
  return a < b ? a : b;
}

u32 min_u32(u32 a, u32 b) {
  return a < b ? a : b;
}

int min_i32(i32 a, i32 b) {
  return a < b ? a : b;
}

u64 min_u64(u64 a, u64 b) {
  return a < b ? a : b;
}

i64 min_i64(i64 a, i64 b) {
  return a < b ? a : b;
}

u16 max_u16(u16 a, u16 b) {
  return a > b ? a : b;
}

i16 max_i16(i16 a, i16 b) {
  return a > b ? a : b;
}

u32 max_u32(u32 a, u32 b) {
  return a > b ? a : b;
}

int max_i32(i32 a, i32 b) {
  return a > b ? a : b;
}

u64 max_u64(u64 a, u64 b) {
  return a > b ? a : b;
}

i64 max_i64(i64 a, i64 b) {
  return a > b ? a : b;
}

float32 sgn_f32(float32 a) {
  float32 b;
  if (a > 0)
    b = 1;
  else if (a < 0)
    b = -1;
  else
    b = 0;
  return b;
}

float64 sgn_f64(float64 a) {
  float32 b;
  if (a > 0)
    b = 1;
  else if (a < 0)
    b = -1;
  else
    b = 0;
  return b;
}

f32 atan2_f32(f32 a, f32 b) {
  return std::atan2(a, b);
}

f64 atan2_f64(f64 a, f64 b) {
  return std::atan2(a, b);
}

f32 pow_f32(f32 a, f32 b) {
  return std::pow(a, b);
}

f64 pow_f64(f64 a, f64 b) {
  return std::pow(a, b);
}

f32 __nv_sgnf(f32 x) {
  return sgn_f32(x);
}

f64 __nv_sgn(f64 x) {
  return sgn_f64(x);
}

struct PhysicalCoordinates {
  i32 val[quadrants_max_num_indices];
};

STRUCT_FIELD_ARRAY(PhysicalCoordinates, val);

#include "quadrants/program/context.h"

STRUCT_FIELD(RuntimeContext, runtime);
STRUCT_FIELD(RuntimeContext, result_buffer)
STRUCT_FIELD(RuntimeContext, cpu_assert_failed)

#include "quadrants/runtime/llvm/runtime_module/atomic.h"

// These structures are accessible by both the LLVM backend and this C++ runtime
// file here (for building complex runtime functions in C++)

// These structs contain some "template parameters"

// Common Attributes
struct StructMeta {
  i32 snode_id;
  std::size_t element_size;
  i64 max_num_elements;

  Ptr (*lookup_element)(Ptr, Ptr, int i);

  Ptr (*from_parent_element)(Ptr);

  u1 (*is_active)(Ptr, Ptr, int i);

  i32 (*get_num_elements)(Ptr, Ptr);

  void (*refine_coordinates)(PhysicalCoordinates *inp_coord, PhysicalCoordinates *refined_coord, int index);

  RuntimeContext *context;
};

STRUCT_FIELD(StructMeta, snode_id)
STRUCT_FIELD(StructMeta, element_size)
STRUCT_FIELD(StructMeta, max_num_elements)
STRUCT_FIELD(StructMeta, get_num_elements);
STRUCT_FIELD(StructMeta, lookup_element);
STRUCT_FIELD(StructMeta, from_parent_element);
STRUCT_FIELD(StructMeta, refine_coordinates);
STRUCT_FIELD(StructMeta, is_active);
STRUCT_FIELD(StructMeta, context);

struct LLVMRuntime;

constexpr bool enable_assert = true;

void quadrants_assert(RuntimeContext *context, u1 test, const char *msg);
void quadrants_assert_runtime(LLVMRuntime *runtime, u1 test, const char *msg);
#define QD_ASSERT_INFO(x, msg) quadrants_assert(context, (u1)(x), msg)
#define QD_ASSERT(x) QD_ASSERT_INFO(x, #x)

void ___stubs___() {
#if ARCH_cuda
  cuda_vprintf(nullptr, nullptr);
  cuda_clock_i64();
#endif
}
}

#if defined(__clang__) || defined(__GNUC__)
template <typename T>
T debug_add(RuntimeContext *ctx, T a, T b, const char *tb) {
  T c;
  if (__builtin_add_overflow(a, b, &c)) {
    quadrants_printf(ctx->runtime, "Addition overflow detected in %s\n", tb);
  }
  return c;
}

template <typename T>
T debug_sub(RuntimeContext *ctx, T a, T b, const char *tb) {
  T c;
  if (__builtin_sub_overflow(a, b, &c)) {
    quadrants_printf(ctx->runtime, "Subtraction overflow detected in %s\n", tb);
  }
  return c;
}

template <typename T>
T debug_mul(RuntimeContext *ctx, T a, T b, const char *tb) {
  T c;
  if (__builtin_mul_overflow(a, b, &c)) {
    quadrants_printf(ctx->runtime, "Multiplication overflow detected in %s\n", tb);
  }
  return c;
}

template <typename T>
T debug_shl(RuntimeContext *ctx, T a, i32 b, const char *tb) {
  T c = a << b;
  if (c >> b != a) {
    quadrants_printf(ctx->runtime, "Shift left overflow detected in %s\n", tb);
  }
  return c;
}

extern "C" {

#define DEFINE_DEBUG_BIN_OP_TY(op, ty)                                    \
  ty debug_##op##_##ty(RuntimeContext *ctx, ty a, ty b, const char *tb) { \
    return debug_##op(ctx, a, b, tb);                                     \
  }

#define DEFINE_DEBUG_BIN_OP(op)   \
  DEFINE_DEBUG_BIN_OP_TY(op, i8)  \
  DEFINE_DEBUG_BIN_OP_TY(op, u8)  \
  DEFINE_DEBUG_BIN_OP_TY(op, i16) \
  DEFINE_DEBUG_BIN_OP_TY(op, u16) \
  DEFINE_DEBUG_BIN_OP_TY(op, i32) \
  DEFINE_DEBUG_BIN_OP_TY(op, u32) \
  DEFINE_DEBUG_BIN_OP_TY(op, i64) \
  DEFINE_DEBUG_BIN_OP_TY(op, u64)

DEFINE_DEBUG_BIN_OP(add)
DEFINE_DEBUG_BIN_OP(sub)
DEFINE_DEBUG_BIN_OP(mul)
DEFINE_DEBUG_BIN_OP(shl)
}
#endif

bool is_power_of_two(uint32 x) {
  return x != 0 && (x & (x - 1)) == 0;
}

/*
A simple list data structure that is infinitely long.
Data are organized in chunks, where each chunk is allocated on demand.
*/

// TODO: there are many i32 types in this class, which may be an issue if there
// are >= 2 ** 31 elements.
struct ListManager {
  static constexpr std::size_t max_num_chunks = 128 * 1024;
  Ptr chunks[max_num_chunks];
  std::size_t element_size{0};
  std::size_t max_num_elements_per_chunk;
  i32 log2chunk_num_elements;
  i32 lock;
  i32 num_elements;
  LLVMRuntime *runtime;

  ListManager(LLVMRuntime *runtime, std::size_t element_size, std::size_t num_elements_per_chunk)
      : element_size(element_size), max_num_elements_per_chunk(num_elements_per_chunk), runtime(runtime) {
    quadrants_assert_runtime(runtime, is_power_of_two(max_num_elements_per_chunk),
                             "max_num_elements_per_chunk must be POT.");
    lock = 0;
    num_elements = 0;
    log2chunk_num_elements = quadrants::log2int(num_elements_per_chunk);
  }

  void append(void *data_ptr);

  i32 reserve_new_element() {
    auto i = atomic_add_i32(&num_elements, 1);
    auto chunk_id = i >> log2chunk_num_elements;
    touch_chunk(chunk_id);
    return i;
  }

  template <typename T>
  void push_back(const T &t) {
    this->append((void *)&t);
  }

  Ptr allocate();

  void touch_chunk(int chunk_id);

  i32 get_num_active_chunks() {
    i32 counter = 0;
    for (int i = 0; i < max_num_chunks; i++) {
      counter += (chunks[i] != nullptr);
    }
    return counter;
  }

  void clear() {
    num_elements = 0;
  }

  void resize(i32 n) {
    num_elements = n;
  }

  Ptr get_element_ptr(i32 i) {
    return chunks[i >> log2chunk_num_elements] + element_size * (i & ((1 << log2chunk_num_elements) - 1));
  }

  template <typename T>
  T &get(i32 i) {
    return *(T *)get_element_ptr(i);
  }

  Ptr touch_and_get(i32 i) {
    touch_chunk(i >> log2chunk_num_elements);
    return get_element_ptr(i);
  }

  i32 size() {
    return num_elements;
  }

  i32 ptr2index(Ptr ptr) {
    auto chunk_size = max_num_elements_per_chunk * element_size;
    for (int i = 0; i < max_num_chunks; i++) {
      quadrants_assert_runtime(runtime, chunks[i] != nullptr, "ptr not found.");
      if (chunks[i] <= ptr && ptr < chunks[i] + chunk_size) {
        return (i << log2chunk_num_elements) + i32((ptr - chunks[i]) / element_size);
      }
    }
    return -1;
  }
};

extern "C" {

struct Element {
  Ptr element;
  int loop_bounds[2];
  PhysicalCoordinates pcoord;
};

STRUCT_FIELD(Element, element);
STRUCT_FIELD(Element, pcoord);
STRUCT_FIELD_ARRAY(Element, loop_bounds);

struct RandState {
  u32 x;
  u32 y;
  u32 z;
  u32 w;
  i32 lock;
};

void initialize_rand_state(RandState *state, u32 i) {
  state->x = 123456789 * i * 1000000007;
  state->y = 362436069;
  state->z = 521288629;
  state->w = 88675123;
  state->lock = 0;
}
}

struct NodeManager;

struct PreallocatedMemoryChunk {
  Ptr preallocated_head = nullptr;
  Ptr preallocated_tail = nullptr;
  std::size_t preallocated_size = 0;
};

struct LLVMRuntime {
  PreallocatedMemoryChunk runtime_objects_chunk;
  PreallocatedMemoryChunk runtime_memory_chunk;

  host_allocator_type host_allocator;
  assert_failed_type assert_failed;
  host_printf_type host_printf;
  host_vsnprintf_type host_vsnprintf;
  Ptr memory_pool;

  Ptr roots[kMaxNumSnodeTreesLlvm];
  size_t root_mem_sizes[kMaxNumSnodeTreesLlvm];

  Ptr thread_pool;
  parallel_for_type parallel_for;
  ListManager *element_lists[quadrants_max_num_snodes];
  NodeManager *node_allocators[quadrants_max_num_snodes];
  Ptr ambient_elements[quadrants_max_num_snodes];
  Ptr temporaries;
  RandState *rand_states;

  // Cross backend (CPU, CUDA, AMDGPU) runtime memory allocation
  Ptr allocate_aligned(PreallocatedMemoryChunk &memory_chunk,
                       std::size_t size,
                       std::size_t alignment,
                       bool request = false);

  // Allocate from preallocated memory (CUDA, AMDGPU)
  Ptr allocate_from_reserved_memory(PreallocatedMemoryChunk &memory_chunk, std::size_t size, std::size_t alignment);
  Ptr profiler;
  void (*profiler_start)(Ptr, Ptr);
  void (*profiler_stop)(Ptr);

  char error_message_template[quadrants_error_message_max_length];
  uint64 error_message_arguments[quadrants_error_message_max_num_arguments];
  i32 error_message_lock = 0;
  i64 error_code = 0;
  // Dedicated overflow signal. Pointer to a 64-bit slot in pinned host memory (CUDA `cuMemAllocHost_v2`,
  // HIP `hipHostMalloc`; CPU plain malloc; on this struct stored as the device-mapped address obtained via
  // `cuMemHostGetDevicePointer` / HIP equivalent). The kernel-side `stack_push` writes via a system-wide
  // atomic OR through this pointer; the host polls the corresponding host-side pointer (cached separately on
  // `LlvmRuntimeExecutor`) without any DtoH or sync drain. Required hardware capability is system-scope
  // atomics on host-mapped memory: NVIDIA Compute Capability 6.0+ (Pascal+, 2016) and AMD GFX9+ (Vega+, 2017),
  // matching the existing pinned-host-scratch H2D-async pattern in `llvm_adstack_lazy_claim.cpp`. Separate from
  // `error_code` so assertions (which set error_code=1, gated on `compile_config.debug`) do not leak through the
  // always-on overflow poll. nullptr until `materialize_runtime` initialises it; nullptr-guarded in `stack_push`.
  i64 *adstack_overflow_flag_dev_ptr = nullptr;

  // Pinned-host slot recording the Program-assigned u32 identity of the FIRST adstack-sizing-info whose
  // overflow path fired this Quadrants entry window. Codegen emits a `cmpxchg(0, id)` immediately after the
  // OR-1 on `adstack_overflow_flag_dev_ptr`; only the first overflowing thread's id sticks, subsequent
  // threads' cmpxchg fails harmlessly. Host reads the slot during the raise to produce a diagnostic that
  // names the offending kernel / task / stack via `Program::adstack_sizing_info_registry_`. nullptr until
  // `materialize_runtime` allocates the slot. Lives on the same UVA-mapped pinned host page as the flag
  // above so a single `mem_alloc_host` call covers both.
  i64 *adstack_overflow_task_id_dev_ptr = nullptr;

  // Combined-heap fields. The codegen single-heap path reads these directly; the split-heap path leaves them untouched
  // and uses the per-kind fields below. Kept for backward compatibility with kernels that have not yet migrated to the
  // split layout (no codegen-side opt-in), so existing AdStack* tests stay byte-identical.
  Ptr adstack_heap_buffer = nullptr;
  u64 adstack_heap_size = 0;
  u64 adstack_per_thread_stride = 0;

  // Split-heap fields. Float allocas (`AdStackAllocaStmt::ret_type == f32`) live in `adstack_heap_buffer_float`,
  // addressed by `row_id_var * adstack_per_thread_stride_float + float_offset_within_slice`; the row claim happens
  // lazily at the float Lowest Common Ancestor (LCA) block via an atomic-add into
  // `adstack_row_counters[task_id_in_kernel]`. Int / u1 allocas live in `adstack_heap_buffer_int`, addressed by
  // `linear_thread_idx * adstack_per_thread_stride_int + int_offset_within_slice` (eager per-thread layout, no row
  // claim). Splitting is what lets the host shrink the float heap to `effective_rows * stride_float` (where
  // `effective_rows` is the count of threads passing the captured `bound_expr` gate) instead of `num_threads *
  // stride_total`. Each buffer is host-owned and grown via the device allocator before each launch; the host caches the
  // field-of-LLVMRuntime pointers via `runtime_get_adstack_heap_field_ptrs` and subsequent grows write through those
  // cached pointers.
  Ptr adstack_heap_buffer_float = nullptr;
  u64 adstack_heap_size_float = 0;
  Ptr adstack_heap_buffer_int = nullptr;
  u64 adstack_heap_size_int = 0;
  u64 adstack_per_thread_stride_float = 0;
  u64 adstack_per_thread_stride_int = 0;

  // Per-launch adstack metadata buffers. Populated by the host right before each kernel launch from the
  // `AdStackAllocaStmt::size_expr` host evaluator, consumed inside the kernel by the LLVM codegen base-address and
  // push-overflow math. `adstack_offsets[stack_id]` is the byte offset within the per-thread slice of the appropriate
  // kind (the codegen selects the slice at compile time based on `AdStackAllocaStmt::ret_type`), and
  // `adstack_max_sizes[stack_id]` is the per-launch max-size. Both arrays live in device-visible memory.
  u64 *adstack_offsets = nullptr;
  u64 *adstack_max_sizes = nullptr;

  // Per-task atomic counter array (`u32[num_tasks_in_kernel]`) for the lazy LCA-block float-heap row claim. Each task
  // with a float adstack atomic-adds 1 into its slot at the LCA block; the returned value becomes the thread's
  // `row_id_var`. Host clears slots before the launch and reads them back after to drive the grow-on-demand path on
  // `adstack_heap_buffer_float`. Sized for the largest kernel observed; lives with the LLVMRuntime for its full
  // lifetime.
  u32 *adstack_row_counters = nullptr;
  u64 adstack_row_counters_capacity = 0;

  // Per-task captured row capacity (`u32[num_tasks_in_kernel]`) consumed by the codegen-emitted defense-in-depth bounds
  // check at the float LCA-block claim site. For tasks where the host reducer published a per-task count, the slot
  // holds that count; for every other task, the slot holds UINT32_MAX so the bounds check is inert by construction.
  // Same lifetime / sizing pattern as `adstack_row_counters`.
  u32 *adstack_bound_row_capacities = nullptr;
  u64 adstack_bound_row_capacities_capacity = 0;

  Ptr result_buffer;
  i32 allocator_lock;

  i32 num_rand_states;

  i64 total_requested_memory;

  template <typename T>
  void set_result(std::size_t i, T t) {
    static_assert(sizeof(T) <= sizeof(uint64));
    ((u64 *)result_buffer)[i] = quadrants_union_cast_with_different_sizes<uint64>(t);
  }

  template <typename T, typename... Args>
  T *create(Args &&...args) {
    auto ptr = (T *)allocate_aligned(runtime_memory_chunk, sizeof(T), 4096, true /*request*/);
    new (ptr) T(std::forward<Args>(args)...);
    return ptr;
  }
};

// TODO: are these necessary?
STRUCT_FIELD_ARRAY(LLVMRuntime, element_lists);
STRUCT_FIELD_ARRAY(LLVMRuntime, node_allocators);
STRUCT_FIELD_ARRAY(LLVMRuntime, roots);
STRUCT_FIELD_ARRAY(LLVMRuntime, root_mem_sizes);
STRUCT_FIELD(LLVMRuntime, temporaries);
STRUCT_FIELD(LLVMRuntime, assert_failed);
STRUCT_FIELD(LLVMRuntime, host_printf);
STRUCT_FIELD(LLVMRuntime, host_vsnprintf);
STRUCT_FIELD(LLVMRuntime, profiler);
STRUCT_FIELD(LLVMRuntime, profiler_start);
STRUCT_FIELD(LLVMRuntime, profiler_stop);
STRUCT_FIELD(LLVMRuntime, adstack_heap_buffer);
STRUCT_FIELD(LLVMRuntime, adstack_heap_size);
STRUCT_FIELD(LLVMRuntime, adstack_per_thread_stride);
STRUCT_FIELD(LLVMRuntime, adstack_heap_buffer_float);
STRUCT_FIELD(LLVMRuntime, adstack_heap_size_float);
STRUCT_FIELD(LLVMRuntime, adstack_heap_buffer_int);
STRUCT_FIELD(LLVMRuntime, adstack_heap_size_int);
STRUCT_FIELD(LLVMRuntime, adstack_per_thread_stride_float);
STRUCT_FIELD(LLVMRuntime, adstack_per_thread_stride_int);
STRUCT_FIELD(LLVMRuntime, adstack_offsets);
STRUCT_FIELD(LLVMRuntime, adstack_max_sizes);
STRUCT_FIELD(LLVMRuntime, adstack_row_counters);
STRUCT_FIELD(LLVMRuntime, adstack_bound_row_capacities);
STRUCT_FIELD(LLVMRuntime, adstack_overflow_flag_dev_ptr);
STRUCT_FIELD(LLVMRuntime, adstack_overflow_task_id_dev_ptr);

// NodeManager of node S (hash, pointer) managers the memory allocation of S_ch
// It makes use of three ListManagers.
struct NodeManager {
  LLVMRuntime *runtime;
  i32 lock;

  i32 element_size;
  i32 chunk_num_elements;
  i32 free_list_used;

  ListManager *free_list, *recycled_list, *data_list;
  i32 recycle_list_size_backup;

  using list_data_type = i32;

  NodeManager(LLVMRuntime *runtime, i32 element_size, i32 chunk_num_elements = -1)
      : runtime(runtime), element_size(element_size) {
    // 128K elements per chunk, by default
    if (chunk_num_elements == -1) {
      chunk_num_elements = 128 * 1024;
    }
    // Maximum chunk size = 128 MB
    while (chunk_num_elements > 1 && (uint64)chunk_num_elements * element_size > 128UL * 1024 * 1024) {
      chunk_num_elements /= 2;
    }
    this->chunk_num_elements = chunk_num_elements;
    free_list_used = 0;
    free_list = runtime->create<ListManager>(runtime, sizeof(list_data_type), chunk_num_elements);
    recycled_list = runtime->create<ListManager>(runtime, sizeof(list_data_type), chunk_num_elements);
    data_list = runtime->create<ListManager>(runtime, element_size, chunk_num_elements);
  }

  Ptr allocate() {
    int old_cursor = atomic_add_i32(&free_list_used, 1);
    i32 l;
    if (old_cursor >= free_list->size()) {
      // running out of free list. allocate new.
      l = data_list->reserve_new_element();
    } else {
      // reuse
      l = free_list->get<list_data_type>(old_cursor);
    }
    return data_list->get_element_ptr(l);
  }

  i32 locate(Ptr ptr) {
    return data_list->ptr2index(ptr);
  }

  void recycle(Ptr ptr) {
    auto index = locate(ptr);
    recycled_list->append(&index);
  }

  void gc_serial() {
    // compact free list
    for (int i = free_list_used; i < free_list->size(); i++) {
      free_list->get<list_data_type>(i - free_list_used) = free_list->get<list_data_type>(i);
    }
    const i32 num_unused = max_i32(free_list->size() - free_list_used, 0);
    free_list_used = 0;
    free_list->resize(num_unused);

    // zero-fill recycled and push to free list
    for (int i = 0; i < recycled_list->size(); i++) {
      auto idx = recycled_list->get<list_data_type>(i);
      auto ptr = data_list->get_element_ptr(idx);
      std::memset(ptr, 0, element_size);
      free_list->push_back(idx);
    }
    recycled_list->clear();
  }
};

extern "C" {

void RuntimeContext_store_result(RuntimeContext *ctx, u64 ret, u32 idx) {
  ctx->result_buffer[quadrants_result_buffer_ret_value_id + idx] = ret;
}

void LLVMRuntime_profiler_start(LLVMRuntime *runtime, Ptr kernel_name) {
  runtime->profiler_start(runtime->profiler, kernel_name);
}

void LLVMRuntime_profiler_stop(LLVMRuntime *runtime) {
  runtime->profiler_stop(runtime->profiler);
}

Ptr get_temporary_pointer(LLVMRuntime *runtime, u64 offset) {
  return runtime->temporaries + offset;
}

// Stashes `runtime->temporaries` into the result buffer so the host-side executor can read it back via
// `fetch_result<void *>(quadrants_result_buffer_ret_value_id, ...)`. Written as a dedicated `runtime_`-prefixed
// helper because only functions with that prefix are marked as `.entry` kernels by `init_runtime_module` on
// CUDA and survive the eliminate-unused pass on AMDGPU; the STRUCT_FIELD-generated `LLVMRuntime_get_temporaries`
// getter is not callable via `runtime_jit->call` on GPU backends.
void runtime_get_temporaries_ptr(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_ret_value_id, runtime->temporaries);
}

// Writes the addresses of `runtime->adstack_heap_buffer` and `runtime->adstack_heap_size` into the result buffer so the
// host-side executor can cache them. With those cached device pointers the host grows the heap by issuing two simple
// `memcpy_host_to_device` writes - no per-grow kernel launch for the setters, which sidesteps any questions about
// AMDGPU kernel calling convention on the auto-generated STRUCT_FIELD setters vs the hand-written `runtime_*` wrappers.
// Writes the addresses of the legacy combined-heap fields into the result buffer so the host caches them and then
// issues per-launch grows via `memcpy_host_to_device` to the cached pointers. Returns two addresses: combined-heap-ptr,
// combined-heap-size. The split-heap path uses a separate getter below.
void runtime_get_adstack_heap_field_ptrs(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_ret_value_id, (u64)(void *)&runtime->adstack_heap_buffer);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 1, (u64)(void *)&runtime->adstack_heap_size);
}

// Per-kind heap field getters for the split-heap path. Returns four addresses in fixed slot order: float-buffer-ptr,
// float-size, int-buffer-ptr, int-size.
void runtime_get_adstack_split_heap_field_ptrs(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_ret_value_id, (u64)(void *)&runtime->adstack_heap_buffer_float);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 1, (u64)(void *)&runtime->adstack_heap_size_float);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 2, (u64)(void *)&runtime->adstack_heap_buffer_int);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 3, (u64)(void *)&runtime->adstack_heap_size_int);
}

// Mirrors `runtime_get_adstack_heap_field_ptrs` for the per-launch metadata fields. The host caches the four returned
// addresses once per program and then publishes new values (combined stride + offsets array pointer + max_sizes array
// pointer + float stride + int stride) before every kernel launch via the same `memcpy_host_to_device` / direct-store
// path used for the heap buffers. Slots 0/1/2 keep the legacy ordering (combined-stride, offsets, max_sizes) so any
// host code that has not migrated still works; slots 3/4 are the new per-kind strides.
void runtime_get_adstack_metadata_field_ptrs(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_ret_value_id, (u64)(void *)&runtime->adstack_per_thread_stride);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 1, (u64)(void *)&runtime->adstack_offsets);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 2, (u64)(void *)&runtime->adstack_max_sizes);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 3, (u64)(void *)&runtime->adstack_per_thread_stride_float);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 4, (u64)(void *)&runtime->adstack_per_thread_stride_int);
}

// Writes the addresses of the per-task lazy-claim counter and bound-row-capacity arrays into the result buffer so the
// host caches them once. The arrays themselves are device-resident; the host publishes the array pointers via
// `memcpy_host_to_device` to the cached field addresses whenever the per-task slot count grows beyond the prior
// allocation.
void runtime_get_adstack_lazy_claim_field_ptrs(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_ret_value_id, (u64)(void *)&runtime->adstack_row_counters);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 1, (u64)(void *)&runtime->adstack_bound_row_capacities);
}

// Device-resident adstack SizeExpr interpreter. Runs on whatever backend the LLVM runtime JIT-compiles this
// bitcode to: a plain C function call on CPU, a single-thread kernel launch on CUDA / AMDGPU. The bytecode buffer
// layout is defined by `quadrants/ir/adstack_size_expr_device.h` and produced host-side by
// `encode_adstack_size_expr_device_bytecode` immediately before this call.
//
// For every alloca slot the interpreter walks its tree (recursive descent over node indices that point strictly
// backwards) and writes:
//   - `runtime->adstack_max_sizes[i]` = `clamp(tree_value, 1, max_size_compile_time)` if the tree is non-empty,
//     else `max_size_compile_time`. The compile-time cap is the structural upper bound the pre-pass proved, so
//     the clamp only ever tightens against a buggy tree evaluation; the `max(_, 1)` preserves the "always room
//     for one push" invariant the runtime's `stack_push` relies on.
//   - `runtime->adstack_offsets[i]` = cumulative byte offset inside the per-thread slice.
//   - `runtime->adstack_per_thread_stride` = final running sum (after last alloca).
// The host reads back `adstack_per_thread_stride` via the cached field pointer to size the heap with
// `ensure_adstack_heap`; the offsets / max_sizes arrays stay device-resident and feed the main kernel directly.
//
// Ndarray element access (`ExternalTensorRead`) reads `ctx->arg_buffer` at the `arg_buffer_offset` encoded into
// the node to fetch the data pointer, then indexes by the linear offset computed from the node's indices. There
// is no `array_ptrs` map on device; the host-side encoder has already resolved `arg_id -> arg_buffer_offset`
// through the kernel's `args_type` struct layout.
//
// Recursion bounded by tree depth (typically <10 for observed reverse-mode kernels, <30 worst case). The
// bound-variable scope is kept in a fixed-size array indexed by `var_id`; the host encoder dense-remaps each
// tree's `var_id`s into `[0, kDeviceBoundVarCap)` before emitting bytecode and hard-errors above the cap, so
// `values[var_id]` is always in bounds here.

namespace {

constexpr int kDeviceBoundVarCap = quadrants::lang::kAdStackSizeExprDeviceMaxBoundVars;

struct DeviceEvalScope {
  // Bound-var lookup by `var_id`. Unbound slots are sentinelled by the caller before the interpreter enters the
  // subtree; walking the code paths that read `values[vid]` without a matching `MaxOverRange` bind would be a
  // pre-pass bug. The interpreter does not validate - on GPU backends we cannot afford a host-style assert from
  // device code, so a buggy tree is caught through wrong max_size values and an overflow at `stack_push` rather
  // than a fatal trap here.
  i64 values[kDeviceBoundVarCap];
};

i64 device_load_element(const char *data_ptr, i64 linear, i32 prim_dt) {
  // Enum values mirror `PrimitiveTypeID` in `quadrants/inc/data_type.inc.h` (f16=0, f32=1, f64=2, i8=3, i16=4,
  // i32=5, i64=6, u1=7, u8=8, u16=9, u32=10, u64=11). The pre-pass only emits integer reads (the adstack-size
  // grammar rejects float-typed reads at build_value_expr), so we only decode the integer types here.
  switch (prim_dt) {
    case 3:  // i8
      return (i64) reinterpret_cast<const i8 *>(data_ptr)[linear];
    case 4:  // i16
      return (i64) reinterpret_cast<const i16 *>(data_ptr)[linear];
    case 5:  // i32
      return (i64) reinterpret_cast<const i32 *>(data_ptr)[linear];
    case 6:  // i64
      return reinterpret_cast<const i64 *>(data_ptr)[linear];
    case 8:  // u8
      return (i64) reinterpret_cast<const u8 *>(data_ptr)[linear];
    case 9:  // u16
      return (i64) reinterpret_cast<const u16 *>(data_ptr)[linear];
    case 10:  // u32
      return (i64) reinterpret_cast<const u32 *>(data_ptr)[linear];
    case 11:  // u64
      return (i64) reinterpret_cast<const u64 *>(data_ptr)[linear];
    default:
      return 0;  // unreachable: encoder rejects other types
  }
}

i64 device_eval_node(const quadrants::lang::AdStackSizeExprDeviceNode *nodes,
                     const i32 *indices,
                     i32 node_idx,
                     DeviceEvalScope *scope,
                     const char *arg_buffer) {
  const auto &node = nodes[node_idx];
  using K = quadrants::lang::AdStackSizeExprDeviceKind;
  switch (static_cast<K>(node.kind)) {
    case K::kConst:
      return node.const_value;
    case K::kAdd:
      return device_eval_node(nodes, indices, node.operand_a, scope, arg_buffer) +
             device_eval_node(nodes, indices, node.operand_b, scope, arg_buffer);
    case K::kSub: {
      // Match the host evaluator: clamp negative trip counts to zero so an underflowed `end - begin` doesn't
      // poison a surrounding `Mul` / `MaxOverRange` product.
      i64 lhs = device_eval_node(nodes, indices, node.operand_a, scope, arg_buffer);
      i64 rhs = device_eval_node(nodes, indices, node.operand_b, scope, arg_buffer);
      i64 diff = lhs - rhs;
      return diff > 0 ? diff : 0;
    }
    case K::kMul:
      return device_eval_node(nodes, indices, node.operand_a, scope, arg_buffer) *
             device_eval_node(nodes, indices, node.operand_b, scope, arg_buffer);
    case K::kMax: {
      i64 lhs = device_eval_node(nodes, indices, node.operand_a, scope, arg_buffer);
      i64 rhs = device_eval_node(nodes, indices, node.operand_b, scope, arg_buffer);
      return lhs > rhs ? lhs : rhs;
    }
    case K::kMaxOverRange: {
      i64 begin = device_eval_node(nodes, indices, node.operand_a, scope, arg_buffer);
      i64 end = device_eval_node(nodes, indices, node.operand_b, scope, arg_buffer);
      // Mirror of the host evaluator's iteration guard (see `adstack_size_expr_eval.cpp::evaluate_node`).
      // A range of several million would stall the sizer launch for seconds; anything that wide is almost
      // certainly a pre-pass bug. Hard-stop via quadrants_assert so the failure surfaces at qd.sync() with
      // a clear adstack-sizer attribution rather than a mysterious launch hang.
      constexpr i64 kMaxOverRangeIterations = i64{1} << 24;
      i64 result = 0;
      const i32 var = node.var_id;
      for (i64 i = begin; i < end; ++i) {
        if (i - begin > kMaxOverRangeIterations) {
          break;  // see host evaluator's note; a sibling assertion in the host path will have fired first.
        }
        if (var >= 0 && var < kDeviceBoundVarCap) {
          scope->values[var] = i;
        }
        i64 v = device_eval_node(nodes, indices, node.body_node_idx, scope, arg_buffer);
        if (v > result)
          result = v;
      }
      return result;
    }
    case K::kBoundVariable: {
      const i32 var = node.var_id;
      if (var >= 0 && var < kDeviceBoundVarCap)
        return scope->values[var];
      return 0;
    }
    case K::kExternalTensorRead: {
      // `data_ptr_slot = *(void **)(arg_buffer + arg_buffer_offset)`: read the ndarray's data pointer out of the
      // kernel arg buffer at the offset the host encoder precomputed via `args_type->get_element_offset`. This
      // replaces the host evaluator's `ctx->array_ptrs` map lookup with a straight field read that the device
      // can perform without reaching for a std::unordered_map.
      auto data_ptr_raw = *reinterpret_cast<const char *const *>(arg_buffer + node.arg_buffer_offset);
      // Indices encoded as `[idx_a_raw, elem_stride_a]` pairs per axis, matching `kFieldLoad`'s layout. The
      // host encoder in `adstack_size_expr_eval.cpp` pre-computes the C-order element strides from the
      // launch context's ndarray shape; a 1-D read collapses to `elem_stride = 1` and recovers the original
      // stride-1 sum. The multi-axis case is what this fix unblocks: without the per-axis multiply a 2-D
      // `a[i, j]` read would land on `a_flat[i + j]` instead of `a_flat[i * shape[1] + j]`, silently
      // under-bounding the sizer and tripping `Adstack overflow` at `qd.sync()`.
      i64 linear = 0;
      for (i32 k = 0; k < node.indices_count; ++k) {
        const i32 raw = indices[node.indices_offset + 2 * k];
        const i32 elem_stride = indices[node.indices_offset + 2 * k + 1];
        i64 v = 0;
        if (raw >= 0) {
          v = raw;
        } else {
          const i32 var = -(raw + 1);
          if (var >= 0 && var < kDeviceBoundVarCap)
            v = scope->values[var];
        }
        linear += v * static_cast<i64>(elem_stride);
      }
      return device_load_element(data_ptr_raw, linear, node.prim_dt);
    }
    case K::kFieldLoad:
      // The LLVM encoder always host-folds `FieldLoad` leaves (via `SNodeRwAccessorsBank`) before emitting
      // device bytecode, so the interpreter never sees `kFieldLoad`. It is reserved for the SPIR-V sizer
      // shader's PSB read path. Return zero rather than asserting (this runtime-module compiles to LLVM
      // bitcode with no host-assert facility) so a mis-emitted tree surfaces downstream as a wrong-`max_size`
      // adstack overflow at `qd.sync()` rather than silently UB here.
      return 0;
  }
  return 0;
}

}  // namespace

// Per-arch reducer counterpart to the SPIR-V `adstack_bound_reducer_shader.cpp` compute kernel: a single-thread serial
// function that walks the captured gating ndarray over `[0, length)`, evaluates the comparison + polarity at each
// thread index, and writes the gate-passing count into `runtime->adstack_bound_row_capacities[task_index]`. The
// codegen-emitted clamp at the float LCA-block claim site reads that slot back, so on backends that have a working
// reducer the bounds clamp activates per task and a future commit can size the float heap from the count instead of the
// dispatched-threads worst case.
//
// Single-thread execution is intentional: dispatching this as a parallel kernel would need a separate JIT-compiled
// compute kernel with atomic-add semantics per arch (the SPIR-V path emits a parallel reducer; LLVM's runtime functions
// go through `runtime_jit->call` which runs serially - on CUDA / AMDGPU it is a 1x1x1 grid kernel launch, on CPU a
// regular function call). For typical iteration bounds (a few hundred thousand on the largest reverse-mode kernels), a
// single device thread completes the count in well under a millisecond per task; that cost is dominated by the actual
// main kernel anyway.
//
// Both ndarray-backed and SNode-backed sources are dispatched through this function: the params blob's
// `field_source_is_snode` flag selects between reading the gating field through the kernel arg buffer (ndarray) or
// through `runtime->roots[snode_root_id]` (SNode), and the comparison + count loop is shared.
void runtime_eval_static_bound_count(LLVMRuntime *runtime, RuntimeContext *ctx, Ptr params_blob) {
  using quadrants::lang::kLlvmReducerCmpEq;
  using quadrants::lang::kLlvmReducerCmpGe;
  using quadrants::lang::kLlvmReducerCmpGt;
  using quadrants::lang::kLlvmReducerCmpLe;
  using quadrants::lang::kLlvmReducerCmpLt;
  using quadrants::lang::kLlvmReducerCmpNe;
  using quadrants::lang::LlvmAdStackBoundReducerDeviceParams;

  const auto *params = reinterpret_cast<const LlvmAdStackBoundReducerDeviceParams *>(params_blob);

  // Resolve the gating field's per-cell pointer + stride based on `field_source_is_snode`. The two source shapes share
  // the comparison + count loop below; only the per-`gid` element load differs.
  //   - ndarray (`field_source_is_snode == 0`): walk `data_ptr[i]` where `data_ptr` is reconstructed from the
  //     kernel arg buffer at `arg_word_offset` (u64 stored across two adjacent u32 words). The element stride is
  //     `sizeof(float)` / `sizeof(i32)` since ndarray data is densely packed by index.
  //   - SNode (`field_source_is_snode == 1`): walk `runtime->roots[snode_root_id] + snode_byte_base_offset +
  //     gid * snode_byte_cell_stride`. The base byte offset and cell stride were pre-resolved at codegen time by
  //     walking the SNode descriptor chain. Mirrors the SPIR-V reducer's `field_source_is_snode` branch.
  const char *field_base = nullptr;
  u32 element_stride_bytes = 0u;
  if (params->field_source_is_snode != 0u) {
    field_base = reinterpret_cast<const char *>(runtime->roots[params->snode_root_id]) + params->snode_byte_base_offset;
    element_stride_bytes = params->snode_byte_cell_stride;
  } else {
    const u32 *arg_buffer_u32 = reinterpret_cast<const u32 *>(ctx->arg_buffer);
    const u64 lo = static_cast<u64>(arg_buffer_u32[params->arg_word_offset]);
    const u64 hi = static_cast<u64>(arg_buffer_u32[params->arg_word_offset + 1]);
    field_base = reinterpret_cast<const char *>(lo | (hi << 32));
    // f32 / i32 share the 4-byte ndarray stride; f64 needs 8 bytes per cell.
    element_stride_bytes = (params->field_dtype_is_float != 0u && params->field_dtype_is_double != 0u)
                               ? 8u
                               : static_cast<u32>(sizeof(u32));
  }

  u32 count = 0;
  if (params->field_dtype_is_float != 0u && params->field_dtype_is_double != 0u) {
    // f64 path: reassemble the 64-bit threshold from the two u32 halves the host packed into the params blob, bitcast
    // to double, then walk the source ndarray as `double *`. f64 thresholds keep the user's full f64 precision;
    // narrowing to f32 here would risk a wrong count on gates whose threshold sits within an f32 representable gap.
    double threshold;
    u64 bits64 = static_cast<u64>(params->threshold_bits) | (static_cast<u64>(params->threshold_bits_high) << 32);
    __builtin_memcpy(&threshold, &bits64, sizeof(double));
    for (u32 i = 0; i < params->length; ++i) {
      const double v = *reinterpret_cast<const double *>(field_base + (u64)i * element_stride_bytes);
      bool match;
      switch (params->cmp_op) {
        case kLlvmReducerCmpLt:
          match = v < threshold;
          break;
        case kLlvmReducerCmpLe:
          match = v <= threshold;
          break;
        case kLlvmReducerCmpGt:
          match = v > threshold;
          break;
        case kLlvmReducerCmpGe:
          match = v >= threshold;
          break;
        case kLlvmReducerCmpEq:
          match = v == threshold;
          break;
        case kLlvmReducerCmpNe:
          match = v != threshold;
          break;
        default:
          match = false;
          break;
      }
      if ((params->polarity != 0u) ? match : !match) {
        ++count;
      }
    }
  } else if (params->field_dtype_is_float != 0u) {
    float threshold;
    {
      // Bitcast the threshold's u32 storage back to f32. memcpy keeps the LLVM IR semantics-clean (no aliasing) and
      // compiles to a single load on every supported arch.
      u32 bits = params->threshold_bits;
      __builtin_memcpy(&threshold, &bits, sizeof(float));
    }
    for (u32 i = 0; i < params->length; ++i) {
      const float v = *reinterpret_cast<const float *>(field_base + (u64)i * element_stride_bytes);
      bool match;
      switch (params->cmp_op) {
        case kLlvmReducerCmpLt:
          match = v < threshold;
          break;
        case kLlvmReducerCmpLe:
          match = v <= threshold;
          break;
        case kLlvmReducerCmpGt:
          match = v > threshold;
          break;
        case kLlvmReducerCmpGe:
          match = v >= threshold;
          break;
        case kLlvmReducerCmpEq:
          match = v == threshold;
          break;
        case kLlvmReducerCmpNe:
          match = v != threshold;
          break;
        default:
          match = false;
          break;
      }
      if ((params->polarity != 0u) ? match : !match) {
        ++count;
      }
    }
  } else {
    const i32 threshold = static_cast<i32>(params->threshold_bits);
    for (u32 i = 0; i < params->length; ++i) {
      const i32 v = *reinterpret_cast<const i32 *>(field_base + (u64)i * element_stride_bytes);
      bool match;
      switch (params->cmp_op) {
        case kLlvmReducerCmpLt:
          match = v < threshold;
          break;
        case kLlvmReducerCmpLe:
          match = v <= threshold;
          break;
        case kLlvmReducerCmpGt:
          match = v > threshold;
          break;
        case kLlvmReducerCmpGe:
          match = v >= threshold;
          break;
        case kLlvmReducerCmpEq:
          match = v == threshold;
          break;
        case kLlvmReducerCmpNe:
          match = v != threshold;
          break;
        default:
          match = false;
          break;
      }
      if ((params->polarity != 0u) ? match : !match) {
        ++count;
      }
    }
  }

  runtime->adstack_bound_row_capacities[params->task_index] = count;
}

void runtime_eval_adstack_size_expr(LLVMRuntime *runtime, RuntimeContext *ctx, Ptr bytecode) {
  // Bytecode layout:
  // [AdStackSizeExprDeviceHeader][stack_headers[n_stacks]][nodes[total_nodes]][indices[total_indices]]. All three
  // arrays live contiguously so the interpreter can index them by offset from the single `bytecode` pointer - the host
  // memcpys the whole blob in one go, and this function runs before any main-kernel dispatch that would stomp
  // `arg_buffer`.
  using quadrants::lang::AdStackSizeExprDeviceHeader;
  using quadrants::lang::AdStackSizeExprDeviceNode;
  using quadrants::lang::AdStackSizeExprDeviceStackHeader;

  const auto *header = reinterpret_cast<const AdStackSizeExprDeviceHeader *>(bytecode);
  const auto *stack_headers = reinterpret_cast<const AdStackSizeExprDeviceStackHeader *>(
      reinterpret_cast<const char *>(bytecode) + sizeof(AdStackSizeExprDeviceHeader));
  const auto *nodes = reinterpret_cast<const AdStackSizeExprDeviceNode *>(
      reinterpret_cast<const char *>(stack_headers) + sizeof(AdStackSizeExprDeviceStackHeader) * header->n_stacks);
  const auto *indices = reinterpret_cast<const i32 *>(reinterpret_cast<const char *>(nodes) +
                                                      sizeof(AdStackSizeExprDeviceNode) * header->total_nodes);

  const char *arg_buffer = ctx->arg_buffer;
  u64 *out_max_sizes = runtime->adstack_max_sizes;
  u64 *out_offsets = runtime->adstack_offsets;

  // Alignment rule copied from `publish_adstack_metadata` in `llvm_runtime_executor.cpp`: each stack's slice ends
  // aligned to 8 bytes so `stack_top_primal`'s `stack + sizeof(u64) + idx * 2 * element_size` math stays aligned
  // for every element type the IR may emit.
  auto align_up_8 = [](u64 n) -> u64 { return (n + 7u) & ~(u64)7u; };

  DeviceEvalScope scope;
  for (i32 k = 0; k < kDeviceBoundVarCap; ++k)
    scope.values[k] = 0;

  // Per-kind running offsets for the unconditional split-heap codegen path. Float allocas address via `row_id_var *
  // stride_float + float_offset_within_float_slice`; int / u1 allocas address via `linear_tid * stride_int +
  // int_offset_within_int_slice`. `out_offsets[i]` therefore must be the byte offset within the per-kind slice, not
  // within a combined slice (the codegen and the host-eval branch in `publish_adstack_metadata` both pick the per-kind
  // base + stride at the use site, so a combined offset would alias float and int slots for any kernel with mixed-kind
  // adstacks). The combined running offset is also tracked for the legacy `runtime->adstack_per_thread_stride` field
  // that offline-cache-loaded kernels predating the split read; on freshly-compiled kernels nothing dereferences it.
  u64 running_offset_combined = 0;
  u64 running_offset_float = 0;
  u64 running_offset_int = 0;
  for (u32 i = 0; i < header->n_stacks; ++i) {
    const auto &sh = stack_headers[i];
    u64 max_size;
    if (sh.root_node_idx < 0) {
      // No symbolic bound captured (offline-cache-hit with `size_exprs` dropped) - use the compile-time bound.
      max_size = sh.max_size_compile_time > 0 ? sh.max_size_compile_time : 1;
    } else {
      i64 v = device_eval_node(nodes, indices, sh.root_node_idx, &scope, arg_buffer);
      // Floor at 1 to match the host evaluator (`evaluate_adstack_size_expr`); a tree that evaluates to 0 or negative
      // leaves one slot reserved so the heap base address is still valid and any spurious push surfaces as an overflow
      // rather than a zero-slice alias. Do NOT clamp upward against `max_size_compile_time`: the compile-time seed is a
      // conservative placeholder for offline-cache fallback, NOT a proven upper bound. Clamping `v` against it would
      // silently truncate correct per-launch values and trigger overflow at the next sync; the SizeExpr evaluator is
      // the authoritative source for the per-launch capacity, and any push past `v` is the real overflow.
      if (v < 1)
        v = 1;
      max_size = static_cast<u64>(v);
    }
    out_max_sizes[i] = max_size;
    const u64 step = align_up_8(sizeof(i64) + (u64)sh.entry_size_bytes * max_size);
    if (sh.heap_kind == 0u) {
      out_offsets[i] = running_offset_float;
      running_offset_float += step;
    } else {
      out_offsets[i] = running_offset_int;
      running_offset_int += step;
    }
    running_offset_combined += step;
  }

  // Mirror the host-eval branch's contract (`llvm_runtime_executor.cpp::publish_adstack_metadata`): the legacy
  // `adstack_per_thread_stride` field publishes `stride_int_bytes` on both paths so any offline-cache-loaded kernel
  // that still reads it observes a consistent value. Earlier drafts published the combined `stride_float + stride_int`
  // here, which diverged from the host-eval branch on any kernel with at least one ExternalTensorRead-leaf SizeExpr
  // (the `use_host_eval=false` gate).
  (void)running_offset_combined;
  runtime->adstack_per_thread_stride = running_offset_int;
  runtime->adstack_per_thread_stride_float = running_offset_float;
  runtime->adstack_per_thread_stride_int = running_offset_int;
}

void runtime_retrieve_and_reset_error_code(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_error_id, runtime->error_code);
  runtime->error_code = 0;
}

// Publish the device-mapped address of the pinned host slot the host allocated for the adstack overflow flag.
// Called once at materialise_runtime time after the host allocates the slot via `cuMemAllocHost_v2` / `hipHostMalloc`
// / plain malloc and obtains the device-mapped address (CUDA `cuMemHostGetDevicePointer` / HIP equivalent / identity
// on CPU). Subsequent kernel-side `stack_push` reads this pointer to write the overflow signal; the host polls the
// host-side address directly without involving any JIT helper.
void runtime_set_adstack_overflow_flag_dev_ptr(LLVMRuntime *runtime, void *dev_ptr) {
  runtime->adstack_overflow_flag_dev_ptr = (i64 *)dev_ptr;
}

// Companion to `runtime_set_adstack_overflow_flag_dev_ptr`. Called once at materialise_runtime alongside the
// flag setter. The task-id slot lives on the same pinned host page so a single allocation backs both. Codegen
// emits a `cmpxchg(0, baked_id)` against this pointer at the lazy-claim overflow path; only the first
// overflowing thread's id sticks. Host reads the slot during the raise to look up the offending kernel /
// task in `Program::adstack_sizing_info_registry_`.
void runtime_set_adstack_overflow_task_id_dev_ptr(LLVMRuntime *runtime, void *dev_ptr) {
  runtime->adstack_overflow_task_id_dev_ptr = (i64 *)dev_ptr;
}

void runtime_retrieve_error_message(LLVMRuntime *runtime, int i) {
  runtime->set_result(quadrants_result_buffer_error_id, runtime->error_message_template[i]);
}

void runtime_retrieve_error_message_argument(LLVMRuntime *runtime, int argument_id) {
  runtime->set_result(quadrants_result_buffer_error_id, runtime->error_message_arguments[argument_id]);
}

void runtime_ListManager_get_num_active_chunks(LLVMRuntime *runtime, ListManager *list_manager) {
  runtime->set_result(quadrants_result_buffer_runtime_query_id, list_manager->get_num_active_chunks());
}

RUNTIME_STRUCT_FIELD_ARRAY(LLVMRuntime, node_allocators);
RUNTIME_STRUCT_FIELD_ARRAY(LLVMRuntime, element_lists);
// Host-side runtime-query getter for `runtime->roots[snode_root_id]`. The CPU bound-reducer host evaluator in
// `LlvmRuntimeExecutor::publish_per_task_bound_count_cpu` uses this to walk SNode-backed gating fields (`field_base =
// roots[id] + snode_byte_base_offset`); the device-side reducer reads the same array directly from device code, so no
// runtime_query wrapper is needed there.
RUNTIME_STRUCT_FIELD_ARRAY(LLVMRuntime, roots);
RUNTIME_STRUCT_FIELD(LLVMRuntime, total_requested_memory);

RUNTIME_STRUCT_FIELD(NodeManager, free_list);
RUNTIME_STRUCT_FIELD(NodeManager, recycled_list);
RUNTIME_STRUCT_FIELD(NodeManager, data_list);
RUNTIME_STRUCT_FIELD(NodeManager, free_list_used);

RUNTIME_STRUCT_FIELD(ListManager, num_elements);
RUNTIME_STRUCT_FIELD(ListManager, max_num_elements_per_chunk);
RUNTIME_STRUCT_FIELD(ListManager, element_size);

void quadrants_assert(RuntimeContext *context, u1 test, const char *msg) {
  quadrants_assert_runtime(context->runtime, test, msg);
}

void quadrants_assert_format(LLVMRuntime *runtime, u1 test, const char *format, int num_arguments, uint64 *arguments) {
#ifdef ARCH_amdgpu
  // TODO: find out why error with mark_force_no_inline
  //  llvm::SDValue llvm::SelectionDAG::getNode(unsigned int, const llvm::SDLoc
  //  &, llvm::EVT, llvm::SDValue, const llvm::SDNodeFlags): Assertion
  //  `VT.getSizeInBits() == Operand.getValueSizeInBits() && "Cannot BITCAST
  //  between types of different sizes!"' failed.
#else
  mark_force_no_inline();
#endif
  if (!enable_assert || test != 0)
    return;
  if (!runtime->error_code) {
    locked_task(&runtime->error_message_lock, [&] {
      if (!runtime->error_code) {
        runtime->error_code = 1;  // Assertion failure

        memset(runtime->error_message_template, 0, quadrants_error_message_max_length);
        memcpy(runtime->error_message_template, format,
               std::min(quadrants_strlen(format), quadrants_error_message_max_length - 1));
        for (int i = 0; i < num_arguments; i++) {
          runtime->error_message_arguments[i] = arguments[i];
        }
      }
    });
  }
#if ARCH_cuda
  // Kill this CUDA thread.
  asm("exit;");
#elif ARCH_amdgpu
  asm("S_ENDPGM");
  // TODO: properly kill this CPU thread here, considering the containing
  // ThreadPool structure.

  // std::terminate();

  // Note that std::terminate() will throw an signal 6
  // (Aborted), which will be caught by Quadrants's signal handler. The assert
  // failure message will NOT be properly printed since Quadrants exits after
  // receiving that signal. It is better than nothing when debugging the
  // runtime, since otherwise the whole program may crash if the kernel
  // continues after assertion failure.
#endif
}

// Context-aware variant called by bounds-check assertions in JIT'd code.
// Returns 1 when the assertion failed (so the codegen can emit an early
// return), 0 otherwise.  This replaces a previous setjmp/longjmp approach
// that crashed on Windows because JIT'd frames lack SEH unwind tables.
i32 quadrants_assert_format_ctx(RuntimeContext *context,
                                u1 test,
                                const char *format,
                                int num_arguments,
                                uint64 *arguments) {
  quadrants_assert_format(context->runtime, test, format, num_arguments, arguments);
#if !ARCH_cuda && !ARCH_amdgpu
  if (enable_assert && test == 0) {
    context->cpu_assert_failed = 1;
    return 1;
  }
#endif
  return 0;
}

void quadrants_assert_runtime(LLVMRuntime *runtime, u1 test, const char *msg) {
  quadrants_assert_format(runtime, test, msg, 0, nullptr);
}

// [ON HOST] CPU backend
// [ON DEVICE] CUDA/AMDGPU backend
Ptr LLVMRuntime::allocate_aligned(PreallocatedMemoryChunk &memory_chunk,
                                  std::size_t size,
                                  std::size_t alignment,
                                  bool request) {
  if (request)
    atomic_add_i64(&total_requested_memory, size);

  if (memory_chunk.preallocated_size > 0) {
    return allocate_from_reserved_memory(memory_chunk, size, alignment);
  }

  return (Ptr)host_allocator(memory_pool, size, alignment);
}

// [ONLY ON DEVICE] CUDA/AMDGPU backend
Ptr LLVMRuntime::allocate_from_reserved_memory(PreallocatedMemoryChunk &memory_chunk,
                                               std::size_t size,
                                               std::size_t alignment) {
  Ptr ret = nullptr;
  bool success = false;
  locked_task(&allocator_lock, [&] {
    std::size_t preallocated_head = (std::size_t)memory_chunk.preallocated_head;
    std::size_t preallocated_tail = (std::size_t)memory_chunk.preallocated_tail;

    auto alignment_bytes = alignment - 1 - (preallocated_head + alignment - 1) % alignment;
    size += alignment_bytes;
    if (preallocated_head + size <= preallocated_tail) {
      ret = (Ptr)(preallocated_head + alignment_bytes);
      memory_chunk.preallocated_head += size;
      success = true;
    } else {
      success = false;
    }
  });
  if (!success) {
#if ARCH_cuda
    // Here unfortunately we have to rely on a native CUDA assert failure to
    // halt the whole grid. Using a quadrants_assert_runtime will not finish the
    // whole kernel execution immediately.
    __assertfail(
        "Out of CUDA pre-allocated memory.\n"
        "Consider using ti.init(device_memory_fraction=0.9) or "
        "ti.init(device_memory_GB=4) to allocate more"
        " GPU memory",
        "Quadrants JIT", 0, "allocate_from_reserved_memory", 1);
#endif
  }
  quadrants_assert_runtime(this, success, "Out of pre-allocated memory");
  return ret;
}

// External API
// [ON HOST] CPU backend
// [ON DEVICE] CUDA/AMDGPU backend
void runtime_memory_allocate_aligned(LLVMRuntime *runtime, std::size_t size, std::size_t alignment, uint64 *result) {
  *result = (uint64) nullptr;
  *result = quadrants_union_cast_with_different_sizes<uint64>(
      runtime->allocate_aligned(runtime->runtime_memory_chunk, size, alignment));
}

// External API
// [ON HOST] CPU backend
// [ON DEVICE] CUDA/AMDGPU backend
void runtime_get_memory_requirements(Ptr result_buffer, i32 num_rand_states, i32 use_preallocated_buffer) {
  i64 size = 0;

  if (use_preallocated_buffer) {
    size += quadrants::iroundup(i64(sizeof(LLVMRuntime)), quadrants_page_size);
  }

  size += quadrants::iroundup(i64(quadrants_global_tmp_buffer_size), quadrants_page_size);
  size += quadrants::iroundup(i64(sizeof(RandState)) * num_rand_states, quadrants_page_size);

  reinterpret_cast<i64 *>(result_buffer)[0] = size;
}

// External API
// [ON HOST] CPU backend
// [ON DEVICE] CUDA/AMDGPU backend
void runtime_initialize(Ptr result_buffer,
                        Ptr memory_pool,
                        std::size_t preallocated_size,  // Non-zero means use the preallocated buffer
                        Ptr preallocated_buffer,
                        i32 num_rand_states,
                        void *_host_allocator,
                        void *_host_printf,
                        void *_host_vsnprintf) {
  // bootstrap
  auto host_allocator = (host_allocator_type)_host_allocator;
  auto host_printf = (host_printf_type)_host_printf;
  auto host_vsnprintf = (host_vsnprintf_type)_host_vsnprintf;
  LLVMRuntime *runtime = nullptr;
  Ptr preallocated_tail = preallocated_buffer + preallocated_size;
  if (preallocated_size) {
    runtime = (LLVMRuntime *)preallocated_buffer;
    preallocated_buffer += quadrants::iroundup(sizeof(LLVMRuntime), quadrants_page_size);
  } else {
    runtime = (LLVMRuntime *)host_allocator(memory_pool, sizeof(LLVMRuntime), 128);
  }

  PreallocatedMemoryChunk runtime_objects_chunk;
  runtime_objects_chunk.preallocated_size = preallocated_size;
  runtime_objects_chunk.preallocated_head = preallocated_buffer;
  runtime_objects_chunk.preallocated_tail = preallocated_tail;

  runtime->runtime_objects_chunk = std::move(runtime_objects_chunk);

  runtime->result_buffer = result_buffer;
  runtime->set_result(quadrants_result_buffer_ret_value_id, runtime);
  runtime->host_allocator = host_allocator;
  runtime->host_printf = host_printf;
  runtime->host_vsnprintf = host_vsnprintf;
  runtime->memory_pool = memory_pool;

  runtime->total_requested_memory = 0;

  // Zero-init the adstack metadata fields (see AdStackSizingInfo usage): `LLVMRuntime` is allocated from a raw
  // memory pool rather than constructed via `new`, so the C++ default-member-initializers on these fields never
  // run. The host launcher writes real values into them via `publish_adstack_metadata` before dispatching any
  // adstack-bearing kernel, but we still zero them here so an assert-driven read (or a stale cached kernel that
  // runs before any publish) sees well-defined zeros instead of garbage.
  runtime->adstack_heap_buffer = nullptr;
  runtime->adstack_heap_size = 0;
  runtime->adstack_per_thread_stride = 0;
  runtime->adstack_heap_buffer_float = nullptr;
  runtime->adstack_heap_size_float = 0;
  runtime->adstack_heap_buffer_int = nullptr;
  runtime->adstack_heap_size_int = 0;
  runtime->adstack_per_thread_stride_float = 0;
  runtime->adstack_per_thread_stride_int = 0;
  runtime->adstack_offsets = nullptr;
  runtime->adstack_max_sizes = nullptr;
  runtime->adstack_row_counters = nullptr;
  runtime->adstack_row_counters_capacity = 0;
  runtime->adstack_bound_row_capacities = nullptr;
  runtime->adstack_bound_row_capacities_capacity = 0;
  runtime->adstack_overflow_flag_dev_ptr = nullptr;
  runtime->adstack_overflow_task_id_dev_ptr = nullptr;

  runtime->temporaries = (Ptr)runtime->allocate_aligned(runtime->runtime_objects_chunk,
                                                        quadrants_global_tmp_buffer_size, quadrants_page_size);

  runtime->num_rand_states = num_rand_states;
  runtime->rand_states = (RandState *)runtime->allocate_aligned(
      runtime->runtime_objects_chunk, sizeof(RandState) * runtime->num_rand_states, quadrants_page_size);
}

void runtime_initialize_memory(LLVMRuntime *runtime, std::size_t preallocated_size, Ptr preallocated_buffer) {
  if (preallocated_size) {
    runtime->runtime_memory_chunk.preallocated_size = preallocated_size;
    runtime->runtime_memory_chunk.preallocated_head = preallocated_buffer;
    runtime->runtime_memory_chunk.preallocated_tail = preallocated_buffer + preallocated_size;
  }
}

void runtime_initialize_rand_states_cuda(LLVMRuntime *runtime, int starting_rand_state) {
  int i = block_dim() * block_idx() + thread_idx();
  initialize_rand_state(&runtime->rand_states[i], starting_rand_state + i);
}

void runtime_initialize_rand_states_serial(LLVMRuntime *runtime, int starting_rand_state) {
  for (int i = 0; i < runtime->num_rand_states; i++) {
    initialize_rand_state(&runtime->rand_states[i], starting_rand_state + i);
  }
}

void runtime_initialize_snodes(LLVMRuntime *runtime,
                               std::size_t root_size,
                               const int root_id,
                               const int num_snodes,
                               const int snode_tree_id,
                               std::size_t rounded_size,
                               Ptr ptr,
                               bool all_dense) {
  // For Metal runtime, we have to make sure that both the beginning address
  // and the size of the root buffer memory are aligned to page size.
  runtime->root_mem_sizes[snode_tree_id] = rounded_size;
  runtime->roots[snode_tree_id] = ptr;
  // runtime->request_allocate_aligned ready to use
  // initialize the root node element list
  if (all_dense) {
    return;
  }
  for (int i = root_id; i < root_id + num_snodes; i++) {
    // TODO: some SNodes do not actually need an element list.
    runtime->element_lists[i] = runtime->create<ListManager>(runtime, sizeof(Element), 1024 * 64);
  }
  Element elem;
  elem.loop_bounds[0] = 0;
  elem.loop_bounds[1] = 1;
  elem.element = runtime->roots[snode_tree_id];
  for (int i = 0; i < quadrants_max_num_indices; i++) {
    elem.pcoord.val[i] = 0;
  }

  runtime->element_lists[root_id]->append(&elem);
}

void LLVMRuntime_initialize_thread_pool(LLVMRuntime *runtime, void *thread_pool, void *parallel_for) {
  runtime->thread_pool = (Ptr)thread_pool;
  runtime->parallel_for = (parallel_for_type)parallel_for;
}

void runtime_NodeAllocator_initialize(LLVMRuntime *runtime, int snode_id, std::size_t node_size) {
  runtime->node_allocators[snode_id] = runtime->create<NodeManager>(runtime, node_size, 1024 * 16);
}

void runtime_allocate_ambient(LLVMRuntime *runtime, int snode_id, std::size_t size) {
  // Do not use NodeManager for the ambient node since it will never be garbage
  // collected.
  runtime->ambient_elements[snode_id] =
      runtime->allocate_aligned(runtime->runtime_memory_chunk, size, 128, true /*request*/);
}

void mutex_lock_i32(Ptr mutex) {
  while (atomic_exchange_i32((i32 *)mutex, 1) == 1)
    ;
}

void mutex_unlock_i32(Ptr mutex) {
  atomic_exchange_i32((i32 *)mutex, 0);
}

int32 ctlz_i32(i32 val) {
  return 0;
}

int32 cttz_i32(i32 val) {
  return 0;
}

int32 cuda_compute_capability() {
  return 0;
}

int32 cuda_ballot(bool bit) {
  return 0;
}

i32 cuda_shfl_down_sync_i32(u32 mask, i32 val, i32 delta, int width) {
  return 0;
}

i32 cuda_shfl_down_i32(i32 delta, i32 val, int width) {
  return 0;
}

f32 cuda_shfl_down_sync_f32(u32 mask, f32 val, i32 delta, int width) {
  return 0;
}

f32 cuda_shfl_down_f32(i32 delta, f32 val, int width) {
  return 0;
}

i32 cuda_shfl_xor_sync_i32(u32 mask, i32 val, i32 delta, int width) {
  return 0;
}

i32 cuda_shfl_up_sync_i32(u32 mask, i32 val, i32 delta, int width) {
  return 0;
}

f32 cuda_shfl_up_sync_f32(u32 mask, f32 val, i32 delta, int width) {
  return 0;
}

i32 cuda_shfl_sync_i32(u32 mask, i32 val, i32 delta, int width) {
  return 0;
}

f32 cuda_shfl_sync_f32(u32 mask, f32 val, i32 delta, int width) {
  return 0;
}

// Stubs patched to AMDGPU intrinsics at module load (see llvm_context.cpp).
// The bodies are replaced by patch_intrinsic; __builtin_trap guards against
// accidentally calling an unpatched stub.

i32 amdgpu_ds_bpermute(i32 byte_index, i32 value) {
  __builtin_trap();
  return 0;
}

i32 amdgpu_mbcnt_lo(i32 mask, i32 base) {
  __builtin_trap();
  return 0;
}

i32 amdgpu_mbcnt_hi(i32 mask, i32 base) {
  __builtin_trap();
  return 0;
}

i32 amdgpu_lane_id() {
  return amdgpu_mbcnt_hi(-1, amdgpu_mbcnt_lo(-1, 0));
}

i32 amdgpu_shuffle_i32(i32 index, i32 value) {
  return amdgpu_ds_bpermute(index * 4, value);
}

f32 amdgpu_shuffle_f32(i32 index, f32 value) {
  union {
    f32 f;
    i32 i;
  } u;
  u.f = value;
  u.i = amdgpu_shuffle_i32(index, u.i);
  return u.f;
}

i64 amdgpu_shuffle_i64(i32 index, i64 value) {
  i32 lo = (i32)(u64)value;
  i32 hi = (i32)((u64)value >> 32);
  lo = amdgpu_shuffle_i32(index, lo);
  hi = amdgpu_shuffle_i32(index, hi);
  return (i64)(((u64)(u32)hi << 32) | (u64)(u32)lo);
}

f64 amdgpu_shuffle_f64(i32 index, f64 value) {
  union {
    f64 d;
    i64 i;
  } u;
  u.d = value;
  u.i = amdgpu_shuffle_i64(index, u.i);
  return u.d;
}

// FIXME: Currently emulates shuffle_down via ds_bpermute (~50 cycle latency).
// Should be upgraded to use DPP ROW_SHR instructions (~4-12 cycles) for
// reduction-pattern offsets (1, 2, 4, 8, 16).
i32 amdgpu_shuffle_down_i32(i32 offset, i32 value) {
  return amdgpu_ds_bpermute((amdgpu_lane_id() + offset) * 4, value);
}

f32 amdgpu_shuffle_down_f32(i32 offset, f32 value) {
  union {
    f32 f;
    i32 i;
  } u;
  u.f = value;
  u.i = amdgpu_shuffle_down_i32(offset, u.i);
  return u.f;
}

i64 amdgpu_shuffle_down_i64(i32 offset, i64 value) {
  i32 lo = (i32)(u64)value;
  i32 hi = (i32)((u64)value >> 32);
  lo = amdgpu_shuffle_down_i32(offset, lo);
  hi = amdgpu_shuffle_down_i32(offset, hi);
  return (i64)(((u64)(u32)hi << 32) | (u64)(u32)lo);
}

f64 amdgpu_shuffle_down_f64(i32 offset, f64 value) {
  union {
    f64 d;
    i64 i;
  } u;
  u.d = value;
  u.i = amdgpu_shuffle_down_i64(offset, u.i);
  return u.d;
}

i32 cuda_lane_id() {
  return thread_idx() & 31;
}

i32 cuda_shuffle_i32(i32 index, i32 value) {
  return cuda_shfl_sync_i32(0xFFFFFFFF, value, index, 31);
}

f32 cuda_shuffle_f32(i32 index, f32 value) {
  union {
    f32 f;
    i32 i;
  } u;
  u.f = value;
  u.i = cuda_shuffle_i32(index, u.i);
  return u.f;
}

i64 cuda_shuffle_i64(i32 index, i64 value) {
  i32 lo = (i32)(u64)value;
  i32 hi = (i32)((u64)value >> 32);
  lo = cuda_shuffle_i32(index, lo);
  hi = cuda_shuffle_i32(index, hi);
  return (i64)(((u64)(u32)hi << 32) | (u64)(u32)lo);
}

f64 cuda_shuffle_f64(i32 index, f64 value) {
  union {
    f64 d;
    i64 i;
  } u;
  u.d = value;
  u.i = cuda_shuffle_i64(index, u.i);
  return u.d;
}

i32 cuda_shuffle_down_i32(i32 offset, i32 value) {
  return cuda_shfl_down_sync_i32(0xFFFFFFFF, value, offset, 31);
}

f32 cuda_shuffle_down_f32(i32 offset, f32 value) {
  union {
    f32 f;
    i32 i;
  } u;
  u.f = value;
  u.i = cuda_shuffle_down_i32(offset, u.i);
  return u.f;
}

i64 cuda_shuffle_down_i64(i32 offset, i64 value) {
  i32 lo = (i32)(u64)value;
  i32 hi = (i32)((u64)value >> 32);
  lo = cuda_shuffle_down_i32(offset, lo);
  hi = cuda_shuffle_down_i32(offset, hi);
  return (i64)(((u64)(u32)hi << 32) | (u64)(u32)lo);
}

f64 cuda_shuffle_down_f64(i32 offset, f64 value) {
  union {
    f64 d;
    i64 i;
  } u;
  u.d = value;
  u.i = cuda_shuffle_down_i64(offset, u.i);
  return u.d;
}

bool cuda_all_sync(u32 mask, bool bit) {
  return false;
}

int32 cuda_all_sync_i32(u32 mask, int32 predicate) {
  return (int32)cuda_all_sync(mask, (bool)predicate);
}

bool cuda_any_sync(u32 mask, bool bit) {
  return false;
}

int32 cuda_any_sync_i32(u32 mask, int32 predicate) {
  return (int32)cuda_any_sync(mask, (bool)predicate);
}

bool cuda_uni_sync(u32 mask, bool bit) {
  return false;
}

int32 cuda_uni_sync_i32(u32 mask, int32 predicate) {
  return (int32)cuda_uni_sync(mask, (bool)predicate);
}

int32 cuda_ballot_sync(int32 mask, bool bit) {
  return 0;
}

int32 cuda_ballot_i32(int32 predicate) {
  return cuda_ballot_sync(UINT32_MAX, (bool)predicate);
}

int32 cuda_ballot_sync_i32(u32 mask, int32 predicate) {
  return cuda_ballot_sync(mask, (bool)predicate);
}

uint32 cuda_match_any_sync_i32(u32 mask, i32 value) {
  return 0;
}

// The three functions below used to contain PTX inline asm with arch-specific
// register constraints (#if __aarch64__ "=w" / #elif __x86_64__ "=r").
// On AArch64, clang validated the constraints against the host target and
// embedded them in the bitcode. When that bitcode was later linked into an
// NVPTX module, the LLVM NVPTX backend rejected the AArch64 'w' constraint
// ("couldn't allocate output register for constraint 'w'"), crashing kernel
// compilation for any large CUDA kernel that pulled in these symbols.
//
// The fix: keep the bodies as trivial stubs here (the host compiler never
// actually executes them) and let patch_intrinsic() in llvm_context.cpp
// replace them with the corresponding LLVM NVPTX intrinsics at module-init
// time, which is target-correct and architecture-independent.

u32 cuda_match_all_sync_i32(u32 mask, i32 value) {
  return 0;
}

uint32 cuda_match_any_sync_i64(u32 mask, i64 value) {
  return 0;
}

uint32 cuda_active_mask() {
  return 0;
}

void block_barrier() {
}

int32 block_barrier_and_i32(int32 predicate) {
  return 0;
}

int32 block_barrier_or_i32(int32 predicate) {
  return 0;
}

int32 block_barrier_count_i32(int32 predicate) {
  return 0;
}

void warp_barrier(uint32 mask) {
}

void block_mem_fence() {
#ifdef ARCH_amdgpu
  // Workgroup-scope memory fence (no thread convergence). On CUDA this body
  // is replaced by `patch_intrinsic("block_mem_fence", nvvm_membar_cta, ...)`
  // in `llvm_context.cpp`; on AMDGPU there is no LLVM intrinsic to patch
  // through `patch_intrinsic` (the AMDGCN fence intrinsic takes
  // ordering+syncscope arguments), so we emit the right instruction
  // directly via the Clang builtin and let the AMDGCN backend lower it to
  // `s_waitcnt lgkmcnt(0)` etc.
  __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
#endif
}

void grid_mem_fence() {
#ifdef ARCH_amdgpu
  // Device-scope (cross-workgroup) memory fence. CUDA's body gets patched
  // to `nvvm_membar_gl`; AMDGPU emits the agent-scope fence directly, same
  // reasoning as `block_mem_fence` above.
  __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent");
#endif
}

// these trivial functions are needed by the DEFINE_REDUCTION macro
i32 op_add_i32(i32 a, i32 b) {
  return a + b;
}
f32 op_add_f32(f32 a, f32 b) {
  return a + b;
}

i32 op_min_i32(i32 a, i32 b) {
  return std::min(a, b);
}
f32 op_min_f32(f32 a, f32 b) {
  return std::min(a, b);
}

i32 op_max_i32(i32 a, i32 b) {
  return std::max(a, b);
}
f32 op_max_f32(f32 a, f32 b) {
  return std::max(a, b);
}

i32 op_and_i32(i32 a, i32 b) {
  return a & b;
}
i32 op_or_i32(i32 a, i32 b) {
  return a | b;
}
i32 op_xor_i32(i32 a, i32 b) {
  return a ^ b;
}

#define DEFINE_REDUCTION(op, dtype)                                                     \
  dtype warp_reduce_##op##_##dtype(uint32_t mask, dtype val) {                          \
    for (int offset = 16; offset > 0; offset /= 2)                                      \
      val = op_##op##_##dtype(val, cuda_shfl_down_sync_##dtype(mask, val, offset, 31)); \
    return val;                                                                         \
  }                                                                                     \
  dtype reduce_##op##_##dtype(dtype *result, dtype val) {                               \
    uint32_t mask = cuda_active_mask();                                                 \
    if (mask != 0xFFFFFFFF) {                                                           \
      atomic_##op##_##dtype(result, val);                                               \
    } else {                                                                            \
      dtype warp_result = warp_reduce_##op##_##dtype(0xFFFFFFFF, val);                  \
      if ((thread_idx() & (warp_size() - 1)) == 0) {                                    \
        atomic_##op##_##dtype(result, warp_result);                                     \
      }                                                                                 \
    }                                                                                   \
    return val;                                                                         \
  }

DEFINE_REDUCTION(add, i32);
DEFINE_REDUCTION(add, f32);

DEFINE_REDUCTION(min, i32);
DEFINE_REDUCTION(min, f32);

DEFINE_REDUCTION(max, i32);
DEFINE_REDUCTION(max, f32);

DEFINE_REDUCTION(and, i32);
DEFINE_REDUCTION(or, i32);
DEFINE_REDUCTION(xor, i32);

// "Element", "component" are different concepts

void clear_list(LLVMRuntime *runtime, StructMeta *parent, StructMeta *child) {
  auto child_list = runtime->element_lists[child->snode_id];
  child_list->clear();
}

/*
 * The element list of a SNode, maintains pointers to its instances, and
 * instances' parents' coordinates
 */

// For the root node there is only one container,
// therefore we use a special kernel for more parallelism.
void element_listgen_root(LLVMRuntime *runtime, StructMeta *parent, StructMeta *child) {
  // If there's just one element in the parent list, we need to use the blocks
  // (instead of threads) to split the parent container
  auto parent_list = runtime->element_lists[parent->snode_id];
  auto child_list = runtime->element_lists[child->snode_id];
  // Cache the func pointers here for better compiler optimization
  auto parent_lookup_element = parent->lookup_element;
  auto child_get_num_elements = child->get_num_elements;
  auto child_from_parent_element = child->from_parent_element;
#if ARCH_cuda || ARCH_amdgpu
  // All blocks share the only root container, which has only one child
  // container.
  // Each thread processes a subset of the child container for more parallelism.
  int c_start = block_dim() * block_idx() + thread_idx();
  int c_step = grid_dim() * block_dim();
#else
  int c_start = 0;
  int c_step = 1;
#endif
  // Note that the root node has only one container, and the `element`
  // representing that single container has only one 'child':
  // element.loop_bounds[0] = 0 and element.loop_bounds[1] = 1
  // Therefore, compared with element_listgen_nonroot,
  // we need neither `i` to loop over the `elements`, nor `j` to
  // loop over the children.

  auto element = parent_list->get<Element>(0);

  auto ch_element = parent_lookup_element((Ptr)parent, element.element, 0);
  ch_element = child_from_parent_element((Ptr)ch_element);
  auto ch_num_elements = child_get_num_elements((Ptr)child, ch_element);
  auto ch_element_size = std::min(ch_num_elements, quadrants_listgen_max_element_size);

  // Here is a grid-stride loop.
  for (int c = c_start; c * ch_element_size < ch_num_elements; c += c_step) {
    Element elem;
    elem.element = ch_element;
    elem.loop_bounds[0] = c * ch_element_size;
    elem.loop_bounds[1] = std::min((c + 1) * ch_element_size, ch_num_elements);
    // There is no need to refine coordinates for root listgen, since its
    // num_bits is always zero
    elem.pcoord = element.pcoord;
    child_list->append(&elem);
  }
}

void element_listgen_nonroot(LLVMRuntime *runtime, StructMeta *parent, StructMeta *child) {
  auto parent_list = runtime->element_lists[parent->snode_id];
  int num_parent_elements = parent_list->size();
  auto child_list = runtime->element_lists[child->snode_id];
  // Cache the func pointers here for better compiler optimization
  auto parent_refine_coordinates = parent->refine_coordinates;
  auto parent_is_active = parent->is_active;
  auto parent_lookup_element = parent->lookup_element;
  auto child_get_num_elements = child->get_num_elements;
  auto child_from_parent_element = child->from_parent_element;
#if ARCH_cuda || ARCH_amdgpu
  // Each block processes a slice of a parent container
  int i_start = block_idx();
  int i_step = grid_dim();
  // Each thread processes an element of the parent container
  int j_start = thread_idx();
  int j_step = block_dim();
#else
  int i_start = 0;
  int i_step = 1;
  int j_start = 0;
  int j_step = 1;
#endif
  for (int i = i_start; i < num_parent_elements; i += i_step) {
    auto element = parent_list->get<Element>(i);
    int j_lower = element.loop_bounds[0] + j_start;
    int j_higher = element.loop_bounds[1];
    for (int j = j_lower; j < j_higher; j += j_step) {
      PhysicalCoordinates refined_coord;
      parent_refine_coordinates(&element.pcoord, &refined_coord, j);
      if (parent_is_active((Ptr)parent, element.element, j)) {
        auto ch_element = parent_lookup_element((Ptr)parent, element.element, j);
        ch_element = child_from_parent_element((Ptr)ch_element);
        auto ch_num_elements = child_get_num_elements((Ptr)child, ch_element);
        auto ch_element_size = std::min(ch_num_elements, quadrants_listgen_max_element_size);
        for (int ch_lower = 0; ch_lower < ch_num_elements; ch_lower += ch_element_size) {
          Element elem;
          elem.element = ch_element;
          elem.loop_bounds[0] = ch_lower;
          elem.loop_bounds[1] = std::min(ch_lower + ch_element_size, ch_num_elements);
          elem.pcoord = refined_coord;
          child_list->append(&elem);
        }
      }
    }
  }
}

using BlockTask = void(RuntimeContext *, char *, Element *, int, int);

struct cpu_block_task_helper_context {
  RuntimeContext *context;
  BlockTask *task;
  ListManager *list;
  int element_size;
  int element_split;
  std::size_t tls_buffer_size;
};

// TODO: To enforce inlining, we need to create in LLVM a new function that
// calls block_helper and the BLS xlogues, and pass that function to the
// scheduler.

// TODO: TLS should be directly passed to the scheduler, so that it lives
// with the threads (instead of blocks).

void cpu_struct_for_block_helper(void *ctx_, int thread_id, int i) {
  auto ctx = (cpu_block_task_helper_context *)(ctx_);
  int element_id = i / ctx->element_split;
  int part_size = ctx->element_size / ctx->element_split;
  int part_id = i % ctx->element_split;
  auto &e = ctx->list->get<Element>(element_id);
  int lower = e.loop_bounds[0] + part_id * part_size;
  int upper = e.loop_bounds[0] + (part_id + 1) * part_size;
  upper = std::min(upper, e.loop_bounds[1]);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  alignas(8) char tls_buffer[ctx->tls_buffer_size];
#pragma clang diagnostic pop

  RuntimeContext this_thread_context = *ctx->context;
  this_thread_context.cpu_thread_id = thread_id;
  this_thread_context.cpu_assert_failed = 0;

  if (lower < upper) {
    (*ctx->task)(&this_thread_context, tls_buffer, &ctx->list->get<Element>(element_id), lower, upper);
  }
  if (this_thread_context.cpu_assert_failed)
    ctx->context->cpu_assert_failed = 1;
}

void parallel_struct_for(RuntimeContext *context,
                         int snode_id,
                         int element_size,
                         int element_split,
                         BlockTask *task,
                         std::size_t tls_buffer_size,
                         int num_threads) {
  auto list = (context->runtime)->element_lists[snode_id];
  auto list_tail = list->size();
#if ARCH_cuda || ARCH_amdgpu
  int i = block_idx();
  // Note: CUDA requires compile-time constant local array sizes.
  // We use "1" here and modify it during codegen to tls_buffer_size.
  alignas(8) char tls_buffer[1];
  // TODO: refactor element_split more systematically.
  element_split = 1;
  const auto part_size = element_size / element_split;
  while (true) {
    int element_id = i / element_split;
    if (element_id >= list_tail)
      break;
    auto part_id = i % element_split;
    auto &e = list->get<Element>(element_id);
    int lower = e.loop_bounds[0] + part_id * part_size;
    int upper = e.loop_bounds[0] + (part_id + 1) * part_size;
    upper = std::min(upper, e.loop_bounds[1]);
    if (lower < upper)
      task(context, tls_buffer, &list->get<Element>(element_id), lower, upper);
    i += grid_dim();
  }
#else
  cpu_block_task_helper_context ctx;
  ctx.context = context;
  ctx.task = task;
  ctx.list = list;
  ctx.element_size = element_size;
  ctx.element_split = element_split;
  ctx.tls_buffer_size = tls_buffer_size;
  auto runtime = context->runtime;
  runtime->parallel_for(runtime->thread_pool, list_tail * element_split, num_threads, &ctx,
                        cpu_struct_for_block_helper);
#endif
}

using range_for_xlogue = void (*)(RuntimeContext *, /*TLS*/ char *tls_base);
using mesh_for_xlogue = void (*)(RuntimeContext *,
                                 /*TLS*/ char *tls_base,
                                 uint32_t patch_idx);

struct range_task_helper_context {
  RuntimeContext *context;
  range_for_xlogue prologue{nullptr};
  RangeForTaskFunc *body{nullptr};
  range_for_xlogue epilogue{nullptr};
  std::size_t tls_size{1};
  int begin;
  int end;
  int block_size;
  int step;
};

void cpu_parallel_range_for_task(void *range_context, int thread_id, int task_id) {
  auto ctx = *(range_task_helper_context *)range_context;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  alignas(8) char tls_buffer[ctx.tls_size];
#pragma clang diagnostic pop
  auto tls_ptr = &tls_buffer[0];

  RuntimeContext this_thread_context = *ctx.context;
  this_thread_context.cpu_thread_id = thread_id;
  this_thread_context.cpu_assert_failed = 0;

  if (ctx.prologue) {
    ctx.prologue(&this_thread_context, tls_ptr);
    if (this_thread_context.cpu_assert_failed) {
      ctx.context->cpu_assert_failed = 1;
      return;
    }
  }

  if (ctx.step == 1) {
    int block_start = ctx.begin + task_id * ctx.block_size;
    int block_end = std::min(block_start + ctx.block_size, ctx.end);
    for (int i = block_start; i < block_end; i++) {
      ctx.body(&this_thread_context, tls_ptr, i);
      if (this_thread_context.cpu_assert_failed)
        break;
    }
  } else if (ctx.step == -1) {
    int block_start = ctx.end - task_id * ctx.block_size;
    int block_end = std::max(ctx.begin, block_start - ctx.block_size);
    for (int i = block_start - 1; i >= block_end; i--) {
      ctx.body(&this_thread_context, tls_ptr, i);
      if (this_thread_context.cpu_assert_failed)
        break;
    }
  }

  if (!this_thread_context.cpu_assert_failed && ctx.epilogue)
    ctx.epilogue(&this_thread_context, tls_ptr);
  if (this_thread_context.cpu_assert_failed)
    ctx.context->cpu_assert_failed = 1;
}

void cpu_parallel_range_for(RuntimeContext *context,
                            int num_threads,
                            int begin,
                            int end,
                            int step,
                            int block_dim,
                            range_for_xlogue prologue,
                            RangeForTaskFunc *body,
                            range_for_xlogue epilogue,
                            std::size_t tls_size) {
  range_task_helper_context ctx;
  ctx.context = context;
  ctx.prologue = prologue;
  ctx.tls_size = tls_size;
  ctx.body = body;
  ctx.epilogue = epilogue;
  ctx.begin = begin;
  ctx.end = end;
  ctx.step = step;
  if (step != 1 && step != -1) {
    quadrants_printf(context->runtime, "step must not be %d\n", step);
    exit(-1);
  }
  ctx.block_size = block_dim;
  auto runtime = context->runtime;
  runtime->parallel_for(runtime->thread_pool, (end - begin + block_dim - 1) / block_dim, num_threads, &ctx,
                        cpu_parallel_range_for_task);
}

void gpu_parallel_range_for(RuntimeContext *context,
                            int begin,
                            int end,
                            range_for_xlogue prologue,
                            RangeForTaskFunc *func,
                            range_for_xlogue epilogue,
                            const std::size_t tls_size) {
  int idx = thread_idx() + block_dim() * block_idx() + begin;
#ifdef ARCH_amdgpu
  // AMDGPU doesn't support dynamic array
  // TODO: find a better way to set the tls_size (maybe like struct_for
  alignas(8) char tls_buffer[64];
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  alignas(8) char tls_buffer[tls_size];
#pragma clang diagnostic pop
#endif
  auto tls_ptr = &tls_buffer[0];
  if (prologue)
    prologue(context, tls_ptr);
  while (idx < end) {
    func(context, tls_ptr, idx);
    idx += block_dim() * grid_dim();
  }
  if (epilogue)
    epilogue(context, tls_ptr);
}

struct mesh_task_helper_context {
  RuntimeContext *context;
  mesh_for_xlogue prologue{nullptr};
  RangeForTaskFunc *body{nullptr};
  mesh_for_xlogue epilogue{nullptr};
  std::size_t tls_size{1};
  int num_patches;
  int block_size;
};

void cpu_parallel_mesh_for_task(void *range_context, int thread_id, int task_id) {
  auto ctx = *(mesh_task_helper_context *)range_context;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  alignas(8) char tls_buffer[ctx.tls_size];
#pragma clang diagnostic pop
  auto tls_ptr = &tls_buffer[0];

  RuntimeContext this_thread_context = *ctx.context;
  this_thread_context.cpu_thread_id = thread_id;
  this_thread_context.cpu_assert_failed = 0;

  int block_start = task_id * ctx.block_size;
  int block_end = std::min(block_start + ctx.block_size, ctx.num_patches);

  for (int idx = block_start; idx < block_end; idx++) {
    if (ctx.prologue) {
      ctx.prologue(&this_thread_context, tls_ptr, idx);
      if (this_thread_context.cpu_assert_failed)
        break;
    }
    ctx.body(&this_thread_context, tls_ptr, idx);
    if (this_thread_context.cpu_assert_failed)
      break;
    if (ctx.epilogue) {
      ctx.epilogue(&this_thread_context, tls_ptr, idx);
      if (this_thread_context.cpu_assert_failed)
        break;
    }
  }
  if (this_thread_context.cpu_assert_failed)
    ctx.context->cpu_assert_failed = 1;
}

void cpu_parallel_mesh_for(RuntimeContext *context,
                           int num_threads,
                           int num_patches,
                           int block_dim,
                           mesh_for_xlogue prologue,
                           RangeForTaskFunc *body,
                           mesh_for_xlogue epilogue,
                           std::size_t tls_size) {
  mesh_task_helper_context ctx;
  ctx.context = context;
  ctx.prologue = prologue;
  ctx.tls_size = tls_size;
  ctx.body = body;
  ctx.epilogue = epilogue;
  ctx.num_patches = num_patches;
  if (block_dim == 0) {
    // adaptive block dim
    // ensure each thread has at least ~32 tasks for load balancing
    // and each task has at least 512 items to amortize scheduler overhead
    block_dim = std::min(512, std::max(1, num_patches / (num_threads * 32)));
  }
  ctx.block_size = block_dim;
  auto runtime = context->runtime;
  runtime->parallel_for(runtime->thread_pool, (num_patches + block_dim - 1) / block_dim, num_threads, &ctx,
                        cpu_parallel_mesh_for_task);
}

void gpu_parallel_mesh_for(RuntimeContext *context,
                           int num_patches,
                           mesh_for_xlogue prologue,
                           MeshForTaskFunc *func,
                           mesh_for_xlogue epilogue,
                           const std::size_t tls_size) {
#ifdef ARCH_amdgpu
  // AMDGPU doesn't support dynamic array
  // TODO: find a better way to set the tls_size (maybe like struct_for
  alignas(8) char tls_buffer[64];
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  alignas(8) char tls_buffer[tls_size];
#pragma clang diagnostic pop
#endif
  auto tls_ptr = &tls_buffer[0];
  for (int idx = block_idx(); idx < num_patches; idx += grid_dim()) {
    if (prologue)
      prologue(context, tls_ptr, idx);
    func(context, tls_ptr, idx);
    if (epilogue)
      epilogue(context, tls_ptr, idx);
  }
}

i32 linear_thread_idx(RuntimeContext *context) {
#if ARCH_cuda || ARCH_amdgpu
  return block_idx() * block_dim() + thread_idx();
#else
  return context->cpu_thread_id;
#endif
}

#include "node_dense.h"
#include "node_dynamic.h"
#include "node_pointer.h"
#include "node_root.h"
#include "node_bitmasked.h"

void ListManager::touch_chunk(int chunk_id) {
  quadrants_assert_runtime(runtime, chunk_id < max_num_chunks, "List manager out of chunks.");
  if (!chunks[chunk_id]) {
    locked_task(&lock, [&] {
      // may have been allocated during lock contention
      if (!chunks[chunk_id]) {
        grid_mem_fence();
        auto chunk_ptr = runtime->allocate_aligned(runtime->runtime_memory_chunk,
                                                   max_num_elements_per_chunk * element_size, 4096, true /*request*/);
        atomic_exchange_u64((u64 *)&chunks[chunk_id], (u64)chunk_ptr);
      }
    });
  }
}

void ListManager::append(void *data_ptr) {
  auto ptr = allocate();
  std::memcpy(ptr, data_ptr, element_size);
}

Ptr ListManager::allocate() {
  auto i = reserve_new_element();
  return get_element_ptr(i);
}

void node_gc(LLVMRuntime *runtime, int snode_id) {
  runtime->node_allocators[snode_id]->gc_serial();
}

void gc_parallel_impl_0(RuntimeContext *context, NodeManager *allocator) {
  auto free_list = allocator->free_list;
  auto free_list_size = free_list->size();
  auto free_list_used = allocator->free_list_used;
  using T = NodeManager::list_data_type;

  // Move unused elements to the beginning of the free_list
  int i = linear_thread_idx(context);
  if (free_list_used * 2 > free_list_size) {
    // Directly copy. Dst and src does not overlap
    auto items_to_copy = free_list_size - free_list_used;
    while (i < items_to_copy) {
      free_list->get<T>(i) = free_list->get<T>(free_list_used + i);
      i += grid_dim() * block_dim();
    }
  } else {
    // Move only non-overlapping parts
    auto items_to_copy = free_list_used;
    while (i < items_to_copy) {
      free_list->get<T>(i) = free_list->get<T>(free_list_size - items_to_copy + i);
      i += grid_dim() * block_dim();
    }
  }
}

void gc_parallel_0(RuntimeContext *context, int snode_id) {
  LLVMRuntime *runtime = context->runtime;
  gc_parallel_impl_0(context, runtime->node_allocators[snode_id]);
}

void gc_parallel_impl_1(NodeManager *allocator) {
  auto free_list = allocator->free_list;

  const i32 num_unused = max_i32(free_list->size() - allocator->free_list_used, 0);
  free_list->resize(num_unused);

  allocator->free_list_used = 0;
  allocator->recycle_list_size_backup = allocator->recycled_list->size();
  allocator->recycled_list->clear();
}

void gc_parallel_1(RuntimeContext *context, int snode_id) {
  LLVMRuntime *runtime = context->runtime;
  gc_parallel_impl_1(runtime->node_allocators[snode_id]);
}

void gc_parallel_impl_2(NodeManager *allocator) {
  auto elements = allocator->recycle_list_size_backup;
  auto free_list = allocator->free_list;
  auto recycled_list = allocator->recycled_list;
  auto data_list = allocator->data_list;
  auto element_size = allocator->element_size;
  using T = NodeManager::list_data_type;
  auto i = block_idx();
  while (i < elements) {
    auto idx = recycled_list->get<T>(i);
    auto ptr = data_list->get_element_ptr(idx);
    if (thread_idx() == 0) {
      free_list->push_back(idx);
    }
    // memset
    auto ptr_stop = ptr + element_size;
    if ((uint64)ptr % 4 != 0) {
      auto new_ptr = ptr + 4 - (uint64)ptr % 4;
      if (thread_idx() == 0) {
        for (uint8 *p = ptr; p < new_ptr; p++) {
          *p = 0;
        }
      }
      ptr = new_ptr;
    }
    // now ptr is a multiple of 4
    ptr += thread_idx() * sizeof(uint32);
    while (ptr + sizeof(uint32) <= ptr_stop) {
      *(uint32 *)ptr = 0;
      ptr += sizeof(uint32) * block_dim();
    }
    while (ptr < ptr_stop) {
      *ptr = 0;
      ptr++;
    }
    i += grid_dim();
  }
}

void gc_parallel_2(RuntimeContext *context, int snode_id) {
  LLVMRuntime *runtime = context->runtime;
  gc_parallel_impl_2(runtime->node_allocators[snode_id]);
}
}

extern "C" {

u32 rand_u32(RuntimeContext *context) {
  auto state = &((LLVMRuntime *)context->runtime)->rand_states[linear_thread_idx(context)];

  auto &x = state->x;
  auto &y = state->y;
  auto &z = state->z;
  auto &w = state->w;
  auto t = x ^ (x << 11);

  x = y;
  y = z;
  z = w;
  w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));

  return w * 1000000007;  // multiply a prime number here is very necessary -
                          // it decorrelates streams of PRNGs.
}

uint64 rand_u64(RuntimeContext *context) {
  return ((u64)rand_u32(context) << 32) + rand_u32(context);
}

f32 rand_f32(RuntimeContext *context) {
  return (rand_u32(context) >> 8) * (1.0f / 16777216.0f);
}

f64 rand_f64(RuntimeContext *context) {
  return (rand_u64(context) >> 11) * (1.0 / 9007199254740992.0);
}

i32 rand_i32(RuntimeContext *context) {
  return rand_u32(context);
}

i64 rand_i64(RuntimeContext *context) {
  return rand_u64(context);
}
};

struct printf_helper {
  char buffer[1024];
  int tail;

  printf_helper() {
    std::memset(buffer, 0, sizeof(buffer));
    tail = 0;
  }

  void push_back() {
  }

  template <typename... Args, typename T>
  void push_back(T t, Args &&...args) {
    *(T *)&buffer[tail] = t;
    if (tail % sizeof(T) != 0)
      tail += sizeof(T) - tail % sizeof(T);
    // align
    tail += sizeof(T);
    if constexpr ((sizeof...(args)) != 0) {
      push_back(std::forward<Args>(args)...);
    }
  }

  Ptr ptr() {
    return (Ptr) & (buffer[0]);
  }
};

template <typename... Args>
void quadrants_printf(LLVMRuntime *runtime, const char *format, Args &&...args) {
#if ARCH_cuda
  printf_helper helper;
  helper.push_back(std::forward<Args>(args)...);
  cuda_vprintf((Ptr)format, helper.ptr());
#elif ARCH_amdgpu
// TODO: add printf for amdgpu backend
#else
  runtime->host_printf(format, args...);
#endif
}

#include "locked_task.h"

extern "C" {  // local stack operations

// The stack index `n` is clamped on read so that overflow (push past capacity) does not let subsequent pops and
// top-accesses underflow it and index far out of bounds. The corresponding stack_push writes through
// `runtime->adstack_overflow_flag_dev_ptr` (the device-mapped address of a pinned host slot) and skips the
// increment instead of trapping, so the host-side launcher surfaces the failure as a Python exception rather
// than killing the process via __builtin_trap. When n == 0 (pop-after-overflow underflow path) we return a
// pointer to slot 0 - an uninitialized-but-in-bounds slot. The caller will read garbage from it, but the host
// polls the pinned slot at every Quadrants Python entry and raises before any such value reaches user code.
Ptr stack_top_primal(Ptr stack, std::size_t element_size) {
  auto n = *(u64 *)stack;
  std::size_t idx = n > 0 ? n - 1 : 0;
  return stack + sizeof(u64) + idx * 2 * element_size;
}

Ptr stack_top_adjoint(Ptr stack, std::size_t element_size) {
  return stack_top_primal(stack, element_size) + element_size;
}

void stack_init(Ptr stack) {
  *(u64 *)stack = 0;
}

void stack_pop(Ptr stack) {
  auto &n = *(u64 *)stack;
  if (n > 0) {
    n--;
  }
}

void stack_push(LLVMRuntime *runtime,
                Ptr stack,
                size_t max_num_elements,
                std::size_t element_size,
                i64 task_registry_id) {
  u64 &n = *(u64 *)stack;
  if (n + 1 > max_num_elements) {
    // Overflow: the loop has more iterations than the adstack capacity. Skip the push and flip the dedicated
    // overflow flag in pinned host memory. The host polls the pinned slot at every Quadrants Python entry
    // and raises a `QuadrantsAssertionError` with a diagnosis routed through a synchronous sizer that
    // distinguishes a Quadrants bug (pre-pass undercount of the bound) from a user-side mutation that bypassed
    // tracking (DLPack zero-copy is the typical case; the sizer's freshly-computed required size will exceed
    // the cached allocated size in that case).
    //
    // Relaxed atomic ordering: multiple threads can hit this branch concurrently (CPU thread pool, GPU warp
    // divergence) and they all store the same sentinel value, so no inter-thread ordering is required. On
    // CPU this compiles to a naturally-aligned store; on CUDA/AMDGPU device kernels with the pointer aimed
    // at pinned host memory (`cuMemAllocHost_v2` / `hipHostMalloc` with the device-mapped address obtained
    // via `cuMemHostGetDevicePointer` / HIP equivalent) the store is a system-wide atomic on UVA host memory.
    // Available on Compute Capability 6.0+ / GFX9+, the same hardware envelope the existing pinned-host
    // H2D-async pattern in `llvm_adstack_lazy_claim.cpp` already requires.
    //
    // Nullptr-guard: `adstack_overflow_flag_dev_ptr` is nullptr until `materialize_runtime` initialises it.
    // A kernel running before that (a stale cached kernel, a C++-only test) silently no-ops the overflow
    // signal. The runtime cannot raise from device code; this is the safest behavior.
    i64 *flag_ptr = runtime->adstack_overflow_flag_dev_ptr;
    if (flag_ptr != nullptr) {
      __atomic_store_n(flag_ptr, (i64)1, __ATOMIC_RELAXED);
    }
    // Record task identity in the companion pinned-host slot via cmpxchg(0, registry_id) so the host raise
    // site can name the offending kernel + task in the diagnostic message. `task_registry_id == 0` means
    // "not registered" (e.g. a deserialised offline-cache task that has not yet been re-registered); skip
    // the cmpxchg so the slot stays zero and the host falls through to the generic dual-cause message.
    i64 *task_id_ptr = runtime->adstack_overflow_task_id_dev_ptr;
    if (task_id_ptr != nullptr && task_registry_id != 0) {
      i64 expected = 0;
      __atomic_compare_exchange_n(task_id_ptr, &expected, task_registry_id, /*weak=*/false, __ATOMIC_RELAXED,
                                  __ATOMIC_RELAXED);
    }
    return;
  }
  n += 1;
  std::memset(stack_top_primal(stack, element_size), 0, element_size * 2);
}

#include "internal_functions.h"

// TODO: make here less repetitious.
// Original implementation is
// u##N mask = ((((u##N)1 << bits) - 1) << offset);
// When N equals bits equals 32, 32 times of left shifting will be carried on
// which is an undefined behavior.
// see #2096 for more details
#define DEFINE_SET_PARTIAL_BITS(N)                                                                                  \
  void set_mask_b##N(u##N *ptr, u64 mask, u##N value) {                                                             \
    u##N mask_N = (u##N)mask;                                                                                       \
    *ptr = (*ptr & (~mask_N)) | (value & mask);                                                                     \
  }                                                                                                                 \
                                                                                                                    \
  void atomic_set_mask_b##N(u##N *ptr, u64 mask, u##N value) {                                                      \
    u##N mask_N = (u##N)mask;                                                                                       \
    u##N new_value = 0;                                                                                             \
    u##N old_value = *ptr;                                                                                          \
    do {                                                                                                            \
      old_value = *ptr;                                                                                             \
      new_value = (old_value & (~mask_N)) | (value & mask);                                                         \
    } while (!__atomic_compare_exchange(ptr, &old_value, &new_value, true, std::memory_order::memory_order_seq_cst, \
                                        std::memory_order::memory_order_seq_cst));                                  \
  }                                                                                                                 \
                                                                                                                    \
  void set_partial_bits_b##N(u##N *ptr, u32 offset, u32 bits, u##N value) {                                         \
    u##N mask = ((~(u##N)0) << (N - bits)) >> (N - offset - bits);                                                  \
    set_mask_b##N(ptr, mask, value << offset);                                                                      \
  }                                                                                                                 \
                                                                                                                    \
  void atomic_set_partial_bits_b##N(u##N *ptr, u32 offset, u32 bits, u##N value) {                                  \
    u##N mask = ((~(u##N)0) << (N - bits)) >> (N - offset - bits);                                                  \
    atomic_set_mask_b##N(ptr, mask, value << offset);                                                               \
  }                                                                                                                 \
                                                                                                                    \
  u##N atomic_add_partial_bits_b##N(u##N *ptr, u32 offset, u32 bits, u##N value) {                                  \
    u##N mask = ((~(u##N)0) << (N - bits)) >> (N - offset - bits);                                                  \
    u##N new_value = 0;                                                                                             \
    u##N old_value = *ptr;                                                                                          \
    do {                                                                                                            \
      old_value = *ptr;                                                                                             \
      new_value = old_value + (value << offset);                                                                    \
      new_value = (old_value & (~mask)) | (new_value & mask);                                                       \
    } while (!__atomic_compare_exchange(ptr, &old_value, &new_value, true, std::memory_order::memory_order_seq_cst, \
                                        std::memory_order::memory_order_seq_cst));                                  \
    return old_value;                                                                                               \
  }

DEFINE_SET_PARTIAL_BITS(8);
DEFINE_SET_PARTIAL_BITS(16);
DEFINE_SET_PARTIAL_BITS(32);
DEFINE_SET_PARTIAL_BITS(64);

f32 rounding_prepare_f32(f32 f) {
  /* slower (but clearer) version with branching:
  if (f > 0)
    return f + 0.5;
  else
    return f - 0.5;
  */

  // Branch-free implementation: copy the sign bit of "f" to "0.5"
  i32 delta_bits = (quadrants_union_cast<i32>(f) & 0x80000000) | quadrants_union_cast<i32>(0.5f);
  f32 delta = quadrants_union_cast<f32>(delta_bits);
  return f + delta;
}

f64 rounding_prepare_f64(f64 f) {
  // Same as above
  i64 delta_bits = (quadrants_union_cast<i64>(f) & 0x8000000000000000LL) | quadrants_union_cast<i64>(0.5);
  f64 delta = quadrants_union_cast<f64>(delta_bits);
  return f + delta;
}
}

#endif
