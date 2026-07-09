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
#include "quadrants/math/arithmetic.h"
#include "llvm_runtime.h"
#include "adstack_runtime.h"

// In llvm 15, host_printf_type will be saved as ptr instead of ptr of FunctionType. Add dummy function to save function
// type for host_printf_type.
extern "C" void get_func_type_host_printf(const char *, ...) {
}

#if defined(__linux__) && !ARCH_cuda && defined(QD_ARCH_x64)
__asm__(".symver logf,logf@GLIBC_2.2.5");
__asm__(".symver powf,powf@GLIBC_2.2.5");
__asm__(".symver expf,expf@GLIBC_2.2.5");
#endif

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

#include "adstack_runtime.cpp"

void runtime_retrieve_and_reset_error_code(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_error_id, runtime->error_code);
  runtime->error_code = 0;
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

// By-value variants taking up to 8 arguments in registers, called by the bounds-check codegen from a cold block
// reached only on failure. Passing the arguments by value keeps the hot path of checked accesses free of any
// argument buffer (per-site stack allocation) and marshalling stores; 8 covers one index per axis for every
// supported tensor rank, and richer asserts fall back to the buffer variants above.
void quadrants_assert_format_args8(LLVMRuntime *runtime,
                                   const char *format,
                                   int num_arguments,
                                   uint64 arg0,
                                   uint64 arg1,
                                   uint64 arg2,
                                   uint64 arg3,
                                   uint64 arg4,
                                   uint64 arg5,
                                   uint64 arg6,
                                   uint64 arg7) {
  uint64 arguments[8] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7};
  quadrants_assert_format(runtime, 0, format, num_arguments, arguments);
}

i32 quadrants_assert_format_ctx_args8(RuntimeContext *context,
                                      const char *format,
                                      int num_arguments,
                                      uint64 arg0,
                                      uint64 arg1,
                                      uint64 arg2,
                                      uint64 arg3,
                                      uint64 arg4,
                                      uint64 arg5,
                                      uint64 arg6,
                                      uint64 arg7) {
  uint64 arguments[8] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7};
  return quadrants_assert_format_ctx(context, 0, format, num_arguments, arguments);
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
        "Consider using qd.init(device_memory_fraction=0.9) or "
        "qd.init(device_memory_GB=4) to allocate more"
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
//
// `external_rand_states_buffer` is set to non-zero by the GPU host launcher when the rand-states buffer is provided
// by `PersistentRandStateBuffer` (process-lifetime, host-side singleton) rather than carved out of the per-init
// runtime-objects preallocation. The CPU path still leaves it 0 (rand-states are bumped from `runtime_objects_chunk`
// in `runtime_initialize`).
void runtime_get_memory_requirements(Ptr result_buffer,
                                     i32 num_rand_states,
                                     i32 use_preallocated_buffer,
                                     i32 external_rand_states_buffer) {
  i64 size = 0;

  if (use_preallocated_buffer) {
    size += quadrants::iroundup(i64(sizeof(LLVMRuntime)), quadrants_page_size);
  }

  size += quadrants::iroundup(i64(quadrants_global_tmp_buffer_size), quadrants_page_size);
  if (!external_rand_states_buffer) {
    size += quadrants::iroundup(i64(sizeof(RandState)) * num_rand_states, quadrants_page_size);
  }

  reinterpret_cast<i64 *>(result_buffer)[0] = size;
}

// External API
// [ON HOST] CPU backend
// [ON DEVICE] CUDA/AMDGPU backend
//
// Returns the byte size the rand-states buffer would consume if allocated in-line in `runtime_objects_chunk`. The
// host calls this to size the process-lifetime allocation owned by `PersistentRandStateBuffer`.
void runtime_get_rand_states_buffer_size(Ptr result_buffer, i32 num_rand_states) {
  i64 size = quadrants::iroundup(i64(sizeof(RandState)) * num_rand_states, quadrants_page_size);
  reinterpret_cast<i64 *>(result_buffer)[0] = size;
}

// External API
// [ON HOST] CPU backend
// [ON DEVICE] CUDA/AMDGPU backend
//
// `external_rand_states_buffer` is non-null when the host has provided a process-lifetime rand-states allocation (via
// `PersistentRandStateBuffer`). In that case `runtime->rand_states` is bound to that pointer and no rand-states bytes
// are bumped out of `runtime_objects_chunk`. The CPU path passes nullptr and falls through to the in-chunk
// allocation.
void runtime_initialize(Ptr result_buffer,
                        Ptr memory_pool,
                        std::size_t preallocated_size,  // Non-zero means use the preallocated buffer
                        Ptr preallocated_buffer,
                        i32 num_rand_states,
                        void *_host_allocator,
                        void *_host_printf,
                        void *_host_vsnprintf,
                        Ptr external_rand_states_buffer) {
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

  adstack_runtime_zero_init(runtime);

  runtime->temporaries = (Ptr)runtime->allocate_aligned(runtime->runtime_objects_chunk,
                                                        quadrants_global_tmp_buffer_size, quadrants_page_size);

  runtime->num_rand_states = num_rand_states;
  if (external_rand_states_buffer != nullptr) {
    runtime->rand_states = (RandState *)external_rand_states_buffer;
  } else {
    runtime->rand_states = (RandState *)runtime->allocate_aligned(
        runtime->runtime_objects_chunk, sizeof(RandState) * runtime->num_rand_states, quadrants_page_size);
  }
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

// Exchanges a 32-bit value between lanes ``i`` and ``i ^ 32`` in a single instruction. The native instruction
// ``v_permlane64_b32`` is only available on gfx940+ (CDNA3) and gfx11+ (RDNA3+); ``llvm_context.cpp`` detects the
// target at JIT time and patches this stub to either the ``llvm.amdgcn.permlane64`` intrinsic (on supported
// hardware) or an LDS-roundtrip software emulation (on gfx9xx CDNA1/2 and gfx10.x RDNA1/2). The emulation has higher
// latency (LDS store + ``s_waitcnt`` + LDS load -- roughly tens of cycles per call vs. a few for the native swap),
// but produces correct cross-half results on RDNA wave64 emulation hardware. Used by
// ``amdgpu_cross_half_shuffle_i32`` below to repair the cross-half story for ``ds_bpermute``, which is SIMD32-scoped
// on RDNA.
i32 amdgpu_permlane64(i32 value) {
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

i32 amdgpu_ballot_w32(bool bit) {
  __builtin_trap();
  return 0;
}

i64 amdgpu_ballot_w64(bool bit) {
  __builtin_trap();
  return 0;
}

i32 amdgpu_ballot_i32(i32 predicate) {
  return amdgpu_ballot_w32((bool)predicate);
}

i64 amdgpu_ballot_u64(i32 predicate) {
  return amdgpu_ballot_w64((bool)predicate);
}

i32 amdgpu_lane_id() {
  return amdgpu_mbcnt_hi(-1, amdgpu_mbcnt_lo(-1, 0));
}

// Wave64-aware "read ``value`` from lane ``target_lane``" gather for AMDGPU. Shared by every i32 shuffle variant
// (``shuffle`` / ``shuffle_down`` / ``shuffle_up``); the f32 / i64 / f64 wrappers below decompose into i32 calls and
// therefore inherit the wave64 fix for free.
//
// Why this isn't just ``ds_bpermute``:
//
// The AMDGCN ``ds_bpermute_b32`` instruction takes a 5-bit lane index (bits 2-6 of the byte argument), so it can
// only directly address lanes 0-31 -- regardless of which lane is issuing the read and regardless of wavefront
// size. Concretely: on wave64, ``ds_bpermute(target_lane * 4, value)`` returns ``value[target_lane & 31]`` for
// every lane, never reaching the top half of the wavefront for ``target_lane >= 32``.
//
// To repair the top-half case we pair ``ds_bpermute`` with ``llvm.amdgcn.permlane64``, a single-instruction swap
// between lanes ``i`` and ``i ^ 32``. ``permlane64(value)`` exposes the top-half payload at bottom-half lane indices,
// so ``ds_bpermute(byte, permlane64(value))`` effectively reads from lanes 32-63. We always compute both reads and
// select between them branchlessly based on the high bit of ``target_lane``: bit 5 picks the half.
//
// Note this is correct on every AMDGPU target we run on. On CDNA (gfx9xx, gfx940/942) ``ds_bpermute`` could in
// principle directly address all 64 lanes, but because we always mask the byte argument to ``(target_lane & 31) * 4``
// we never test that path -- on both ISAs the byte index is in [0, 128) and only addresses the bottom half. The
// ``permlane64`` swap then supplies the top-half data: on hardware with the native instruction (gfx940+ CDNA3 /
// gfx11+ RDNA3+) this is a single ``v_permlane64_b32``; on older wave64-capable targets (gfx9xx CDNA1/2, gfx10.x
// RDNA1/2) the JIT patches ``amdgpu_permlane64`` to an LDS roundtrip that produces the same result at higher latency
// (see the patching logic in ``llvm_context.cpp``).
//
// OOR target lanes (``target_lane < 0`` or ``target_lane >= 64``): we mask to ``target_lane & 31`` for the byte and
// ``& 32`` for the half-bit. The behaviour for OOR targets is implementation-defined on every backend (CUDA's
// ``__shfl_sync`` also wraps), and the upstream subgroup ops never rely on it -- ``shuffle_up`` / ``shuffle_down``
// have a ``lane_in_group`` predicate at the call site, ``shuffle_xor`` is always in-range for the mask range we
// support, etc. We just need OOR not to crash or corrupt in-range lanes.
i32 amdgpu_cross_half_shuffle_i32(i32 target_lane, i32 value) {
  // Two parallel reads, then a per-lane select. ``permlane64`` is convergent and must execute uniformly across the
  // wave -- lifting it above the select keeps the AMDGPU backend happy and lets it issue exactly one
  // ``v_permlane64_b32``. ``ds_bpermute`` on RDNA wave64 is SIMD32-scoped with a 5-bit address (top half of the wave
  // is unreachable directly), so ``from_self_half`` handles the same-SIMD case and ``from_other_half`` handles the
  // cross-SIMD case via the ``swapped`` payload. On CDNA the wave is one SIMD64 so both reads return the same value
  // and the select is a no-op; we don't try to optimize that out because the dead read is cheap (LLVM CSE may fold
  // it anyway).
  i32 self_lane = amdgpu_lane_id();
  i32 swapped = amdgpu_permlane64(value);
  i32 byte = (target_lane & 31) * 4;
  // ``llvm.amdgcn.ds.bpermute`` is the real hardware ``ds_bpermute_b32`` -- but if LLVM's uniformity analysis decides
  // ``byte`` is uniform across the wave (e.g. ``target_lane`` is a compile-time constant), it sometimes lowers to a
  // ``v_readlane_b32``-style instruction that addresses lanes 0..31 wave-globally rather than SIMD32-locally. On
  // RDNA wave64 that gives the wrong answer for top-half lanes in cross-half reads (lane 32+ would always read from
  // the bottom half of its SIMD instead of swapping in the other SIMD's payload via ``permlane64``). The empty
  // ``+v`` inline asm marks ``byte`` as a VGPR with an opaque write, forcing LLVM to treat it as per-lane and emit a
  // genuine ``ds_bpermute_b32`` -- which on RDNA does SIMD-local addressing, exactly what we need to pair with the
  // ``permlane64`` swap for cross-half traffic. On CDNA the cost is zero (the instruction is the same shape) and on
  // RDNA the cost is also zero (we'd already be issuing a real ``ds_bpermute`` for the per-lane case; this just
  // makes the constant-target case behave the same way).
  //
  // The ``+v`` constraint names the AMDGPU VGPR register class. clang accepts ``v`` as a constraint name on x86
  // (where it historically means an SSE register) and on amdgcn, but rejects it outright on AArch64 -- the asm is
  // parsed against the host's clang target even though the resulting bitcode is later re-targeted to amdgcn at JIT
  // time (see ``llvm_context.cpp`` setting the module triple to ``amdgcn-amd-amdhsa``). The constraint string is
  // preserved verbatim into the IR, so any host whose front-end accepts ``v`` produces bitcode that the AMDGPU
  // backend later reads correctly. Gate on both ``ARCH_amdgpu`` (the runtime is built once per backend, see
  // ``runtime_module/CMakeLists.txt``) and a host-arch allowlist; on AArch64 manylinux builds we drop the fence,
  // which loses the constant-``target_lane`` VGPR hint -- the per-lane case (the common one) still emits a real
  // ``ds_bpermute_b32`` because uniformity analysis sees per-lane inputs.
#if defined(ARCH_amdgpu) && (defined(__x86_64__) || defined(__i386__) || defined(__amdgcn__))
  __asm__ volatile("" : "+v"(byte));
#endif
  i32 from_self_half = amdgpu_ds_bpermute(byte, value);
  i32 from_other_half = amdgpu_ds_bpermute(byte, swapped);
  return ((target_lane ^ self_lane) & 32) ? from_other_half : from_self_half;
}

i32 amdgpu_shuffle_i32(i32 index, i32 value) {
  return amdgpu_cross_half_shuffle_i32(index, value);
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

// FIXME: Currently emulates shuffle_down via the cross-half ``ds_bpermute`` + ``permlane64`` helper (~50-60 cycle
// latency). Should be upgraded to use DPP ROW_SHR instructions (~4-12 cycles) for reduction-pattern offsets (1, 2, 4,
// 8, 16); the cross-half case (offset >= 32) still needs the helper.
i32 amdgpu_shuffle_down_i32(i32 offset, i32 value) {
#if defined(ARCH_amdgpu) && (defined(__x86_64__) || defined(__i386__) || defined(__amdgcn__))
  // DPP fast path for the reduction-step offsets {1,2,4,8}. ``row_shl:N`` shifts data left within each
  // 16-lane row (lane ``i`` reads lane ``i+N``); lanes whose source falls outside the row keep their own
  // value (``r`` is seeded with ``value``), which reduction trees treat as don't-care once the tree has
  // contracted below 16 active lanes. Emitted as inline asm rather than ``__builtin_amdgcn_update_dpp``
  // because runtime.cpp is compiled to bitcode by the host clang (the module is retargeted to amdgcn only
  // at JIT time), so the intrinsic isn't available at compile time; the instruction text is carried
  // verbatim into the IR, same technique as the ``+v`` fence above. ``offset`` is resolved at compile
  // time via constant propagation after inlining, so the dead branches are eliminated. offset 16 crosses
  // a row and >=32 crosses the wave half, so they fall through to the generic helper.
  i32 r = value;
  if (offset == 1) { __asm__ volatile("v_mov_b32_dpp %0, %1 row_shl:1 row_mask:0xf bank_mask:0xf" : "+v"(r) : "v"(value)); return r; }
  if (offset == 2) { __asm__ volatile("v_mov_b32_dpp %0, %1 row_shl:2 row_mask:0xf bank_mask:0xf" : "+v"(r) : "v"(value)); return r; }
  if (offset == 4) { __asm__ volatile("v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0xf" : "+v"(r) : "v"(value)); return r; }
  if (offset == 8) { __asm__ volatile("v_mov_b32_dpp %0, %1 row_shl:8 row_mask:0xf bank_mask:0xf" : "+v"(r) : "v"(value)); return r; }
#endif
  return amdgpu_cross_half_shuffle_i32(amdgpu_lane_id() + offset, value);
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

// Mirrors `amdgpu_shuffle_down`: the cross-half helper is a generic gather, so `shuffle_up` is just `shuffle_down`
// with the source lane index decremented instead of incremented. The same DPP fast-path FIXME applies here too.
i32 amdgpu_shuffle_up_i32(i32 offset, i32 value) {
  // No DPP fast path here (unlike shuffle_down): shuffle_up feeds Hillis-Steele scans, where every lane
  // stays active and reads across 16-lane row boundaries starting at offset 1 (lane 16 reads lane 15).
  // ``row_shr`` would return the boundary lane's own value there instead of the cross-row neighbour, so
  // the scan miscompiles. shuffle_up therefore always uses the generic cross-half helper.
  return amdgpu_cross_half_shuffle_i32(amdgpu_lane_id() - offset, value);
}

f32 amdgpu_shuffle_up_f32(i32 offset, f32 value) {
  union {
    f32 f;
    i32 i;
  } u;
  u.f = value;
  u.i = amdgpu_shuffle_up_i32(offset, u.i);
  return u.f;
}

i64 amdgpu_shuffle_up_i64(i32 offset, i64 value) {
  i32 lo = (i32)(u64)value;
  i32 hi = (i32)((u64)value >> 32);
  lo = amdgpu_shuffle_up_i32(offset, lo);
  hi = amdgpu_shuffle_up_i32(offset, hi);
  return (i64)(((u64)(u32)hi << 32) | (u64)(u32)lo);
}

f64 amdgpu_shuffle_up_f64(i32 offset, f64 value) {
  union {
    f64 d;
    i64 i;
  } u;
  u.d = value;
  u.i = amdgpu_shuffle_up_i64(offset, u.i);
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

// `shfl.sync.up.b32` clamp byte is 0 (no clamp at low boundary), unlike `shfl.sync.down.b32` which uses 0x1f. See
// `qd.simt.warp.shfl_up_*` (which uses width=0) and the NVVM IR / PTX ISA documentation for `shfl.sync.up.b32`.
i32 cuda_shuffle_up_i32(i32 offset, i32 value) {
  return cuda_shfl_up_sync_i32(0xFFFFFFFF, value, offset, 0);
}

f32 cuda_shuffle_up_f32(i32 offset, f32 value) {
  union {
    f32 f;
    i32 i;
  } u;
  u.f = value;
  u.i = cuda_shuffle_up_i32(offset, u.i);
  return u.f;
}

i64 cuda_shuffle_up_i64(i32 offset, i64 value) {
  i32 lo = (i32)(u64)value;
  i32 hi = (i32)((u64)value >> 32);
  lo = cuda_shuffle_up_i32(offset, lo);
  hi = cuda_shuffle_up_i32(offset, hi);
  return (i64)(((u64)(u32)hi << 32) | (u64)(u32)lo);
}

f64 cuda_shuffle_up_f64(i32 offset, f64 value) {
  union {
    f64 d;
    i64 i;
  } u;
  u.d = value;
  u.i = cuda_shuffle_up_i64(offset, u.i);
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
}

void grid_mem_fence() {
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

extern "C" {

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
