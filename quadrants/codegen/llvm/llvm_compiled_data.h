#pragma once

#include <memory>
#include <optional>
#include <unordered_set>

#include "llvm/IR/Module.h"
#include "quadrants/common/serialization.h"
#include "quadrants/ir/adstack_size_expr.h"
#include "quadrants/transforms/static_adstack_analysis.h"

namespace quadrants::lang {

// Sizing information for the per-task adstack heap slice. Populated at codegen time and consumed
// host-side by each kernel launcher before dispatch to grow `LlvmRuntimeExecutor::adstack_heap_` to
// `per_thread_stride * num_threads` bytes via `ensure_adstack_heap`.
//
// `per_thread_stride == 0` means the task has no adstacks; the launcher skips the ensure call.
// Otherwise num_threads is resolved as follows (the launcher applies the same rule on every arch):
//   - If `dynamic_gpu_range_for == false`: use `static_num_threads` directly.
//       - CPU non-serial: the compiler set this to `num_cpu_threads` (slot indexed by `cpu_thread_id`).
//       - CPU serial: the compiler set this to 1.
//       - GPU non-range_for / GPU const-bound range_for: the compiler set this to
//         `grid_dim * block_dim` (tight since codegen caps grid_dim to ceil((end-begin)/block_dim)).
//   - If `dynamic_gpu_range_for == true`: resolve begin and end at launch time and use `end - begin`.
//       - If `begin_offset_bytes >= 0`: memcpy-DtoH 4 bytes from `runtime->temporaries + begin_offset_bytes`.
//       - Else: use `begin_const_value` directly.
//       - Same rule for end.
//     This is the tight sizing for dynamic-bound range_for on GPU - no saturating-grid-dim over-allocation.
// Per-adstack entry in `AdStackSizingInfo::allocas`. One slot per `AdStackAllocaStmt` in the task, indexed by its
// `stack_id`. `offset` is the byte offset of this alloca inside the per-thread slice (a prefix sum of aligned sizes
// of earlier allocas); `max_size_compile_time` is the compile-time bound (used when `size_expr` is absent, e.g.
// after offline-cache load where the symbolic tree is not serialized); `entry_size_bytes` is `2 *
// element_size_in_bytes()` rounded to alignment that matches the runtime `stack_top_primal` math.
struct AdStackAllocaInfo {
  // Heap kind for the dual-heap layout. Float allocas (f32) live on the lazy float heap addressed by `row_id_var
  // * stride_float + offset`; int allocas (i32 / u1) live on the eager int heap addressed by `linear_thread_idx
  // * stride_int + offset`. `offset` is interpreted within the slice of the appropriate kind. `0` = float, `1` = int,
  // matching the SPIR-V `AdStackHeapKind` encoding so the offline cache survives a backend swap.
  enum class HeapKind : int32_t { Float = 0, Int = 1 };
  std::size_t offset{0};
  std::size_t max_size_compile_time{0};
  std::size_t entry_size_bytes{0};
  HeapKind heap_kind{HeapKind::Float};
  QD_IO_DEF(offset, max_size_compile_time, entry_size_bytes, heap_kind);
};

struct AdStackSizingInfo {
  // Combined per-thread stride across all allocas. Equals `per_thread_stride_float + per_thread_stride_int`; kept for
  // backward compatibility with code paths that have not yet been migrated to the split layout.
  std::size_t per_thread_stride{0};
  // Per-thread stride per heap kind. Float stride drives the lazy float heap (addressed by `row_id_var * stride
  // + offset`); int stride drives the eager int heap (addressed by `linear_thread_idx * stride + offset`). Splitting is
  // what lets the host shrink the float heap to `effective_rows * stride_float` (where `effective_rows` is the count of
  // threads passing the captured `bound_expr` gate, when one is recognized) instead of `num_threads * (stride_float +
  // stride_int)`.
  std::size_t per_thread_stride_float{0};
  std::size_t per_thread_stride_int{0};
  std::size_t static_num_threads{0};
  bool dynamic_gpu_range_for{false};
  std::int32_t begin_const_value{0};
  std::int32_t end_const_value{0};
  std::int32_t begin_offset_bytes{-1};
  std::int32_t end_offset_bytes{-1};
  std::vector<AdStackAllocaInfo> allocas;
  // One flat `SerializedSizeExpr` per entry in `allocas`, populated by the codegen pre-scan so the host launcher
  // can evaluate field-load-bounded sizes against the live field state right before each dispatch. The flat post-
  // order form survives the offline cache (an empty `nodes` vector means "no symbolic bound captured", same
  // behaviour as a kernel that Bellman-Ford fully resolved and the launcher only needs `max_size_compile_time`).
  std::vector<SerializedSizeExpr> size_exprs;
  // Captured static gate predicate when the analysis recognized a single recognized `IfStmt` on the LCA-to-root chain.
  // The launcher's per-arch reducer evaluates the predicate over the bound iteration range to shrink the float heap to
  // the actual gate-passing thread count; `nullopt` falls through to dispatched-threads worst-case sizing (no behavior
  // change versus a kernel without this metadata).
  std::optional<StaticAdStackBoundExpr> bound_expr;
  // Identity in `AdStackCache::adstack_sizing_info_registry_`. Assigned by `register_adstack_sizing_info` as
  // `fnv1a32(kernel_name + ":" + task_id_in_kernel)` - a content-stable hash of the unique identity pair below, so the
  // same (kernel_name, task_id_in_kernel) yields the same id across re-compiles, across `Program` lifetimes, and across
  // offline-cache reloads. Baked as an immediate into the codegen-emitted lazy-claim `cmpxchg(0, registry_id)` so the
  // host raise site can name the offending kernel + task in its diagnostic message. `0` is reserved for "not
  // registered" - the codegen short-circuits the cmpxchg in that case. Serialised to the offline cache: a deserialised
  // task carries the same id the codegen produced, matching the immediate baked into its LLVM IR; the runtime
  // re-populates the per-`Program` registry on the first launch via
  // `AdStackCache::ensure_runtime_registry_ids_for_max_reducer`.
  uint32_t registry_id{0};
  // Inputs to the content hash above. Persisted on the per-task adstack metadata (rather than parsed from
  // `OffloadedTask::name`) so the runtime registration call can re-derive the registry entry's diagnostic labels
  // without depending on the function-name format.
  std::string kernel_name;
  int32_t task_id_in_kernel{0};
  // Per-task list of `MaxOverRange` nodes the runtime reduces in parallel via a dedicated max-reducer dispatch (see the
  // max-reducer recognizer). Empty when no captured `size_expr` contains a recognized shape. Each entry references one
  // alloca's `size_expr` by `(stack_id, mor_node_idx)`; the runtime substitutes the dispatched value as a `Const` into
  // the tree before the per-thread sizer walks it.
  std::vector<StaticAdStackMaxReducerSpec> max_reducer_specs;
  QD_IO_DEF(per_thread_stride,
            per_thread_stride_float,
            per_thread_stride_int,
            static_num_threads,
            dynamic_gpu_range_for,
            begin_const_value,
            end_const_value,
            begin_offset_bytes,
            end_offset_bytes,
            allocas,
            size_exprs,
            bound_expr,
            registry_id,
            kernel_name,
            task_id_in_kernel,
            max_reducer_specs);
};

class OffloadedTask {
 public:
  std::string name;
  int block_dim{0};
  int grid_dim{0};
  int dynamic_shared_array_bytes{0};
  int stream_parallel_group_id{0};
  // Nested qd.kernel-as-subgraph (C2). When `is_launch_child` is true this task does not correspond to a generated
  // device function; instead the runtime embeds the `child_call_index`-th child kernel as a subgraph (CUDA/HIP) or
  // launches it sequentially (other backends) at this position in the parent's task stream. The parent supplies one
  // compiled-child descriptor per index at launch time.
  bool is_launch_child{false};
  int child_call_index{-1};
  AdStackSizingInfo ad_stack{};

  // Snode IDs this task writes to (read-modify-write counts as a write). Computed at codegen time
  // by walking the offloaded IR with `gather_snode_read_writes`. Consumed at launch time: each id
  // here bumps `Program::snode_write_gen_[id]` so the per-task adstack metadata cache invalidates
  // whenever a kernel that ran since the cache was recorded mutated a SNode a downstream
  // `size_expr::FieldLoad` may read. Mirrors the SPIR-V `TaskAttributes::snode_writes` field.
  std::vector<int> snode_writes;
  // Argument arg_ids this task writes to (WRITE bit set in `irpass::detect_external_ptr_access_in_task`).
  // Consumed at launch time to bump `Program::ndarray_data_gen_` for the bound DeviceAllocation so
  // the per-task adstack metadata cache invalidates when a kernel that ran since the cache was
  // recorded mutated an ndarray a downstream `size_expr::ExternalTensorRead` reads. Mirrors the
  // SPIR-V `KernelContextAttributes::arr_access` WRITE-bit set, but stored per-task here because
  // LLVM codegen does not aggregate `arr_access` to the kernel level.
  std::vector<int> arr_writes;
  // Argument arg_ids this task reads (READ bit set in `irpass::detect_external_ptr_access_in_task`). Consumed at
  // launch time only on backends with an H2D blit per launch (LLVM-GPU CUDA / AMDGPU `DevAllocType::kNone`) and on
  // CPU LLVM with `DevAllocType::kNone` host args: the data pointer is stable across launches, so the cache key by
  // data ptr cannot detect content mutations the user performed outside Quadrants's tracking. Mirrors the SPIR-V
  // `KernelContextAttributes::arr_access` READ-bit set.
  std::vector<int> arr_reads;

  explicit OffloadedTask(const std::string &name = "",
                         int block_dim = 0,
                         int grid_dim = 0,
                         int dynamic_shared_array_bytes = 0,
                         int stream_parallel_group_id = 0)
      : name(name),
        block_dim(block_dim),
        grid_dim(grid_dim),
        dynamic_shared_array_bytes(dynamic_shared_array_bytes),
        stream_parallel_group_id(stream_parallel_group_id) {};
  QD_IO_DEF(name,
            block_dim,
            grid_dim,
            dynamic_shared_array_bytes,
            stream_parallel_group_id,
            is_launch_child,
            child_call_index,
            ad_stack,
            snode_writes,
            arr_writes,
            arr_reads);
};

struct LLVMCompiledTask {
  std::vector<OffloadedTask> tasks;
  std::unique_ptr<llvm::Module> module{nullptr};
  std::unordered_set<int> used_tree_ids;
  std::unordered_set<int> struct_for_tls_sizes;
  LLVMCompiledTask() = default;
  LLVMCompiledTask(LLVMCompiledTask &&) = default;
  LLVMCompiledTask &operator=(LLVMCompiledTask &&) = default;
  LLVMCompiledTask(std::vector<OffloadedTask> tasks,
                   std::unique_ptr<llvm::Module> module,
                   std::unordered_set<int> used_tree_ids,
                   std::unordered_set<int> struct_for_tls_sizes)
      : tasks(std::move(tasks)),
        module(std::move(module)),
        used_tree_ids(std::move(used_tree_ids)),
        struct_for_tls_sizes(std::move(struct_for_tls_sizes)) {
  }
  LLVMCompiledTask clone() const;
  QD_IO_DEF(tasks);
};

struct LLVMCompiledKernel {
  std::vector<OffloadedTask> tasks;
  std::unique_ptr<llvm::Module> module{nullptr};
  LLVMCompiledKernel() = default;
  LLVMCompiledKernel(LLVMCompiledKernel &&) = default;
  LLVMCompiledKernel &operator=(LLVMCompiledKernel &&) = default;
  LLVMCompiledKernel(std::vector<OffloadedTask> tasks, std::unique_ptr<llvm::Module> module)
      : tasks(std::move(tasks)), module(std::move(module)) {
  }
  LLVMCompiledKernel clone() const;
  QD_IO_DEF(tasks);
};

}  // namespace quadrants::lang
