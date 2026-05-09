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
using RangeForTaskFunc = void(RuntimeContext *, const char *tls, int i);
using MeshForTaskFunc = void(RuntimeContext *, const char *tls, std::uint32_t i);
using parallel_for_type = void (*)(void *thread_pool,
                                   int splits,
                                   int num_desired_threads,
                                   void *context,
                                   void (*func)(void *, int thread_id, int i));

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

  // Per-spec output slot for the max reducer. One i64 per captured `StaticAdStackMaxReducerSpec`, written by
  // `runtime_eval_adstack_max_reduce` during the per-launch dispatch and read by the host launcher to substitute the
  // value as a `Const` into the per-stack `SerializedSizeExpr` tree before any LLVM eval path walks it. Sized / grown
  // by the LlvmRuntimeExecutor lazy-allocate path on the first launch that has captured specs; cleared to INT64_MIN
  // before each dispatch so the running-max sentinel is well-defined when `length == 0` (returns INT64_MIN; the caller
  // floors at 0 + clamps to compile-time).
  i64 *adstack_max_reducer_outputs = nullptr;
  u64 adstack_max_reducer_outputs_capacity = 0;

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
