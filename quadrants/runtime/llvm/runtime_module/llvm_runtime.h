// Shared declarations for the LLVM bitcode runtime translation units (runtime.cpp + adstack_runtime.cpp). Holds the
// `LLVMRuntime` struct memory layout, the cross-TU type aliases, the function-pointer type aliases the runtime takes
// from the host launcher, the `STRUCT_FIELD*` getter macros, and the forward declarations the struct requires. Each TU
// is compiled to its own bitcode and the two are merged via `llvm-link` into the final `runtime_<arch>.bc` consumed by
// `init_runtime_module`. See adstack_runtime.h for the public surface of the adstack subset.

#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

#include "quadrants/inc/constants.h"

// Type aliases used across runtime TUs. Mirrors the original inline definitions in runtime.cpp so existing code that
// uses `i64` / `u64` / `Ptr` / etc. compiles unchanged after the runtime.cpp split.
using int8 = std::int8_t;
using int16 = std::int16_t;
using int32 = std::int32_t;
using int64 = std::int64_t;
using uint1 = bool;
using uint8 = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;
using uint64 = std::uint64_t;
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

using Ptr = uint8 *;

using RuntimeContextArgType = long long;

// Forward declarations. Full definitions live in runtime.cpp - the pointer-only references in `LLVMRuntime` only need
// the names visible here.
struct ListManager;
struct NodeManager;
struct RandState;
struct RuntimeContext;

// Function-pointer types the host launcher fills into the `LLVMRuntime` instance via `LLVMRuntime_initialize`.
using assert_failed_type = void (*)(const char *);
using host_printf_type = void (*)(const char *, ...);
using host_vsnprintf_type = int (*)(char *, std::size_t, const char *, std::va_list);
using host_allocator_type = void *(*)(void *, std::size_t, std::size_t);
using RangeForBodyFn = void(RuntimeContext *, const char *tls, int i);
using MeshForTaskFunc = void(RuntimeContext *, const char *tls, std::uint32_t i);
using parallel_for_type = void (*)(void *thread_pool,
                                   int splits,
                                   int num_desired_threads,
                                   void *context,
                                   void (*task_fn)(void *ctx, int thread_id, int task_id));

// `STRUCT_FIELD` and friends. Generate `extern "C"` getters / setters that the host invokes via the JIT to read /
// write fields by name without taking on the struct layout. Both runtime.cpp and adstack_runtime.cpp use these to
// materialise the per-field accessors on the side of the file the field is owned by.
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
  std::size_t root_mem_sizes[kMaxNumSnodeTreesLlvm];

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

  // qd.init(device_memory_fraction=...) and qd.init(device_memory_GB=...) size only the runtime-memory chunk.
  bool sized_by_device_memory_config(const PreallocatedMemoryChunk &memory_chunk) const;
  Ptr profiler;
  void (*profiler_start)(Ptr, Ptr);
  void (*profiler_stop)(Ptr);

  char error_message_template[quadrants_error_message_max_length];
  uint64 error_message_arguments[quadrants_error_message_max_num_arguments];
  i32 error_message_lock = 0;
  i64 error_code = 0;
  // Reusable measured-stack heap, owned by the host executor.
  Ptr adstack_heap_buffer = nullptr;

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
