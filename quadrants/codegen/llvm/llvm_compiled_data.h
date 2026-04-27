#pragma once

#include <memory>
#include <unordered_set>

#include "llvm/IR/Module.h"
#include "quadrants/common/serialization.h"
#include "quadrants/ir/adstack_size_expr.h"

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
  std::size_t offset{0};
  std::size_t max_size_compile_time{0};
  std::size_t entry_size_bytes{0};
  QD_IO_DEF(offset, max_size_compile_time, entry_size_bytes);
};

struct AdStackSizingInfo {
  std::size_t per_thread_stride{0};
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
  QD_IO_DEF(per_thread_stride,
            static_num_threads,
            dynamic_gpu_range_for,
            begin_const_value,
            end_const_value,
            begin_offset_bytes,
            end_offset_bytes,
            allocas,
            size_exprs);
};

class OffloadedTask {
 public:
  std::string name;
  int block_dim{0};
  int grid_dim{0};
  int dynamic_shared_array_bytes{0};
  AdStackSizingInfo ad_stack{};

  explicit OffloadedTask(const std::string &name = "",
                         int block_dim = 0,
                         int grid_dim = 0,
                         int dynamic_shared_array_bytes = 0)
      : name(name), block_dim(block_dim), grid_dim(grid_dim), dynamic_shared_array_bytes(dynamic_shared_array_bytes) {};
  QD_IO_DEF(name, block_dim, grid_dim, dynamic_shared_array_bytes, ad_stack);
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
