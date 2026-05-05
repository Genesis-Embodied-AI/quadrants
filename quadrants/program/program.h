// Program - Quadrants program execution context

#pragma once

#include <functional>
#include <mutex>
#include <optional>
#include <atomic>
#include <stack>
#include <shared_mutex>
#include <string>
#include <vector>

#define QD_RUNTIME_HOST
#include "quadrants/ir/adstack_size_expr.h"
#include "quadrants/ir/frontend_ir.h"
#include "quadrants/ir/ir.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/ir/snode.h"
#include "quadrants/util/lang_util.h"
#include "quadrants/program/program_impl.h"
#include "quadrants/program/callable.h"
#include "quadrants/program/function.h"
#include "quadrants/program/kernel.h"
#include "quadrants/program/kernel_profiler.h"
#include "quadrants/program/snode_expr_utils.h"
#include "quadrants/program/snode_rw_accessors_bank.h"
#include "quadrants/program/context.h"
#include "quadrants/struct/snode_tree.h"
#include "quadrants/system/threading.h"
#include "quadrants/program/sparse_matrix.h"
#include "quadrants/ir/mesh.h"

namespace quadrants::lang {

class AdStackCache;
class StructCompiler;

/**
 * Note [Backend-specific ProgramImpl]
 * We're working in progress to keep Program class minimal and move all backend
 * specific logic to their corresponding backend ProgramImpls.

 * If you are thinking about exposing/adding attributes/methods to Program
 class,
 * please first think about if it's general for all backends:
 * - If so, please consider adding it to ProgramImpl class first.
 * - Otherwise please add it to a backend-specific ProgramImpl, e.g.
 * LlvmProgramImpl, MetalProgramImpl..
 */

class QD_DLL_EXPORT Program {
 public:
  using Kernel = quadrants::lang::Kernel;

  uint64 *result_buffer{nullptr};  // Note that this result_buffer is used
                                   // only for runtime JIT functions (e.g. `runtime_memory_allocate_aligned`)

  std::vector<std::unique_ptr<Kernel>> kernels;

  std::unique_ptr<KernelProfilerBase> profiler{nullptr};

  // Note: for now we let all Programs share a single TypeFactory for smooth migration. In the future each program
  // should have its own copy.
  static TypeFactory &get_type_factory();

  Program() : Program(default_compile_config.arch) {
  }

  explicit Program(Arch arch);

  ~Program();

  const CompileConfig &compile_config() const {
    return compile_config_;
  }

  struct KernelProfilerQueryResult {
    int counter{0};
    double min{0.0};
    double max{0.0};
    double avg{0.0};
  };

  KernelProfilerQueryResult query_kernel_profile_info(const std::string &name) {
    KernelProfilerQueryResult query_result;
    profiler->query(name, query_result.counter, query_result.min, query_result.max, query_result.avg);
    return query_result;
  }

  void clear_kernel_profile_info() {
    profiler->clear();
  }

  void profiler_start(const std::string &name) {
    profiler->start(name);
  }

  void profiler_stop() {
    profiler->stop();
  }

  KernelProfilerBase *get_profiler() {
    return profiler.get();
  }

  // Drain the backend command queue. Does not raise; for internal use only.
  void synchronize();

  // Drain the queue and raise on any pending user-visible assert (e.g. adstack overflow). Bound to `qd.sync()`.
  void synchronize_and_assert();

  // Per-Quadrants-Python-entry poll for any pending adstack overflow signal. Unlike `synchronize_and_assert`
  // this does NOT drain the queue: it only reads the pinned-host overflow flag (cheap host atomic load) and
  // raises if set. Wired at every host-read entry point (`Ndarray::read`, `SNodeRwAccessorsBank` reads via
  // `Program::launch_kernel`'s built-in poll) so a DLPack-bypass overflow surfaces within one entry of the
  // offending launch even when the user never calls `qd.sync()`.
  void check_adstack_overflow_and_assert();

  StreamSemaphore flush();

  /**
   * Materializes the runtime.
   */
  void materialize_runtime();

  int get_snode_tree_size();

  void dump_cache_data_to_disk();

  const CompiledKernelData *load_fast_cache(const std::string &checksum,
                                            const std::string &kernel_name,
                                            const CompileConfig &compile_config,
                                            const DeviceCapabilityConfig &device_caps);

  Kernel &create_kernel(const std::function<void(Kernel *)> &body,
                        const std::string &name = "",
                        AutodiffMode autodiff_mode = AutodiffMode::kNone);

  Function *create_function(const FunctionKey &func_key);

  CompileResult compile_kernel(const CompileConfig &compile_config,
                               const DeviceCapabilityConfig &device_caps,
                               const Kernel &kernel_def);

  void launch_kernel(const CompiledKernelData &compiled_kernel_data, LaunchContextBuilder &ctx);

  std::size_t get_graph_cache_size() {
    return program_impl_->get_kernel_launcher().get_graph_cache_size();
  }

  bool get_graph_cache_used_on_last_call() {
    return program_impl_->get_kernel_launcher().get_graph_cache_used_on_last_call();
  }

  size_t get_num_offloaded_tasks_on_last_call() const {
    return num_offloaded_tasks_on_last_call_;
  }

  std::size_t get_graph_num_nodes_on_last_call() {
    return program_impl_->get_kernel_launcher().get_graph_num_nodes_on_last_call();
  }

  std::size_t get_graph_total_builds() {
    return program_impl_->get_kernel_launcher().get_graph_total_builds();
  }

  DeviceCapabilityConfig get_device_caps() {
    return program_impl_->get_device_caps();
  }

  Kernel &get_snode_reader(SNode *snode);

  Kernel &get_snode_writer(SNode *snode);

  uint64 fetch_result_uint64(int i);

  template <typename T>
  T fetch_result(int i) {
    return quadrants_union_cast_with_different_sizes<T>(fetch_result_uint64(i));
  }

  Arch get_host_arch() {
    return host_arch();
  }

  float64 get_total_compilation_time() {
    return total_compilation_time_;
  }

  void finalize();

  static int get_kernel_id() {
    static int id = 0;
    QD_ASSERT(id < 100000);
    return id++;
  }

  static int default_block_dim(const CompileConfig &config);

  // Note this method is specific to LlvmProgramImpl, but we keep it here since it's exposed to python.
  void print_memory_profiler_info();

  // Returns zero if the SNode is statically allocated
  std::size_t get_snode_num_dynamically_allocated(SNode *snode);

  inline SNodeFieldMap *get_snode_to_fields() {
    return &snode_to_fields_;
  }

  inline SNodeRwAccessorsBank &get_snode_rw_accessors_bank() {
    return snode_rw_accessors_bank_;
  }

  // Look up an `SNode` in this `Program`'s snode trees by its global `SNode::id`. Used by the host-side adstack
  // size-expression evaluator to rehydrate an `SNode *` from a `snode_id` that survived the offline cache. Linear
  // over all snode trees; called at most once per adstack leaf per kernel launch so the cost is negligible in
  // practice.
  SNode *get_snode_by_id(int snode_id);

  // Adstack-specific caching: per-task adstack-sizer metadata caches (SPIR-V + LLVM-GPU), encoded SPIR-V bytecode
  // cache, per-launch SizeExpr-eval result cache, and per-snode / per-DeviceAllocation generation counters that drive
  // precise invalidation. Defined in `program/adstack_size_expr_eval.h`. Lifecycle matches `Program`.
  AdStackCache &adstack_cache() {
    return *adstack_cache_;
  }

  // Identity registry for adstack-sizer info. Codegen registers each `OffloadedTask::ad_stack` once per
  // kernel compilation and bakes the assigned id as an immediate into the lazy-claim overflow path; on
  // overflow the codegen emits `cmpxchg(0, id)` against the pinned-host task-id slot. The host raise site
  // reads the slot and routes through `diagnose_adstack_overflow_message(id)` to look up the kernel name,
  // task index, and per-stack metadata for an enriched error message. Pointer ownership stays with
  // `OffloadedTask`; entries are added but not removed - the registry size is bounded by the number of
  // adstack-bearing tasks compiled in the program's lifetime, typically dozens.
  // Identity-key for idempotent re-registration. The diagnose path NEVER dereferences this pointer;
  // it stores all size-expression data inline (`size_exprs`) so the entry is self-contained and
  // immune to lifetime issues from the underlying `AdStackSizingInfo` (LLVM) /
  // `AdStackSizingAttribs` (SPIR-V) struct moves. Two structs because LLVM and SPIR-V backends store
  // launch-time strides differently (byte-indexed vs element-indexed) and SPIR-V doesn't carry the
  // dynamic-range-for fields LLVM needs - unifying them would be a backend-pipeline refactor far
  // outside this PR's scope. Inlining `size_exprs` into the registry is the cheapest way to make
  // the registry uniform across both backends without touching the launchers' on-the-hot-path
  // structs.
  struct AdStackSizingInfoEntry {
    const void *identity_key{nullptr};
    std::string kernel_name;
    int task_id_in_kernel{0};
    std::vector<int> allocated_max_sizes;
    std::vector<SerializedSizeExpr> size_exprs;
  };
  uint32_t register_adstack_sizing_info(const void *identity_key,
                                        const std::string &kernel_name,
                                        int task_id_in_kernel,
                                        std::vector<int> allocated_max_sizes,
                                        std::vector<SerializedSizeExpr> size_exprs);
  // Refresh just the `size_exprs` snapshot in an existing registry entry. Used by the LLVM launcher
  // on the first launch of a task whose codegen-time registration could not capture size_exprs (the
  // codegen-time `current_task->ad_stack` had not yet been finalized). No-op for `id == 0` and ids
  // outside the registry range.
  void update_adstack_sizing_info_size_exprs(uint32_t id, std::vector<SerializedSizeExpr> size_exprs);
  // Returns a *copy* of the registry entry (not a pointer into the underlying vector) so the caller can
  // safely hold the data across operations that might trigger another `register_adstack_sizing_info` and
  // grow / reallocate the registry vector (e.g. `evaluate_adstack_size_expr` dispatching a reader kernel
  // that compiles a fresh task). Returns `std::nullopt` for the sentinel id `0` and for out-of-range ids.
  std::optional<AdStackSizingInfoEntry> lookup_adstack_sizing_info(uint32_t id) const;
  // Format a diagnostic message for an overflow signal. `task_id` is the value read from the pinned-host
  // task-id slot (0 if no thread overflowed; otherwise the registry id of the first overflowing task).
  // Returns the dual-cause message body to embed in the `QuadrantsAssertionError` raised at the poll site.
  std::string diagnose_adstack_overflow_message(uint32_t task_id) const;

  /**
   * Destroys a new SNode tree.
   *
   * @param snode_tree The pointer to SNode tree.
   */
  void destroy_snode_tree(SNodeTree *snode_tree);

  /**
   * Adds a new SNode tree.
   *
   * @param root The root of the new SNode tree.
   * @param compile_only Only generates the compiled type
   * @return The pointer to SNode tree.
   *
   * FIXME: compile_only is mostly a hack to make AOT & cross-compilation work.
   * E.g. users who would like to AOT to a specific target backend can do so,
   * even if their platform doesn't support that backend. Unfortunately, the
   * current implementation would leave the backend in a mostly broken state. We
   * need a cleaner design to support both AOT and JIT modes.
   */
  SNodeTree *add_snode_tree(std::unique_ptr<SNode> root, bool compile_only);

  /**
   * Allocates a SNode tree id for a new SNode tree
   *
   * @return The SNode tree id allocated
   *
   * Returns and consumes a free SNode tree id if there is any,
   * Otherwise returns the size of `snode_trees_`
   */
  int allocate_snode_tree_id();

  /**
   * Gets the root of a SNode tree.
   *
   * @param tree_id Index of the SNode tree
   * @return Root of the tree
   */
  SNode *get_snode_root(int tree_id);

  size_t get_field_in_tree_offset(int tree_id, const SNode *child) {
    return program_impl_->get_field_in_tree_offset(tree_id, child);
  }

  DevicePtr get_snode_tree_device_ptr(int tree_id) {
    return program_impl_->get_snode_tree_device_ptr(tree_id);
  }

  Device *get_compute_device() {
    return program_impl_->get_compute_device();
  }

  Device *get_graphics_device() {
    return program_impl_->get_graphics_device();
  }

  // TODO: do we still need result_buffer?
  DeviceAllocation allocate_memory_on_device(std::size_t alloc_size, uint64 *result_buffer) {
    return program_impl_->allocate_memory_on_device(alloc_size, result_buffer);
  }

  Ndarray *create_ndarray(const DataType type,
                          const std::vector<int> &shape,
                          ExternalArrayLayout layout = ExternalArrayLayout::kNull,
                          bool zero_fill = false,
                          const DebugInfo &dbg_info = DebugInfo());

  std::string get_kernel_return_data_layout() {
    return program_impl_->get_kernel_return_data_layout();
  };

  std::string get_kernel_argument_data_layout() {
    return program_impl_->get_kernel_argument_data_layout();
  };

  std::pair<const StructType *, size_t> get_struct_type_with_data_layout(const StructType *old_ty,
                                                                         const std::string &layout);

  void delete_ndarray(Ndarray *ndarray);

  intptr_t get_ndarray_data_ptr_as_int(const Ndarray *ndarray);

  void fill_ndarray_fast_u32(Ndarray *ndarray, uint32_t val);

  Identifier get_next_global_id(const std::string &name = "") {
    return Identifier(global_id_counter_++, name);
  }

  /** Enqueue a custom compute op to the current program execution flow.
   *
   *  @params op The lambda that is invoked to construct the custom compute Op
   *  @params image_refs The image resource references used in this compute Op
   */
  void enqueue_compute_op_lambda(std::function<void(Device *device, CommandList *cmdlist)> op,
                                 const std::vector<ComputeOpImageRef> &image_refs);

  /**
   * TODO(zhanlue): Remove this interface
   *
   * Gets the underlying ProgramImpl object
   *
   * This interface is essentially a hack to temporarily accommodate
   * historical design issues with LLVM backend
   *
   * Please limit its use to LLVM backend only
   */
  ProgramImpl *get_program_impl() {
    QD_ASSERT(arch_uses_llvm(compile_config().arch));
    return program_impl_.get();
  }

  size_t get_num_ndarrays() const {
    return ndarrays_.size();
  }

  // TODO(zhanlue): Move these members and corresponding interfaces to ProgramImpl Ideally, Program should serve as a
  // pure interface class and all the implementations should fall inside ProgramImpl
  //
  // Once we migrated these implementations to ProgramImpl, lower-level objects could store ProgramImpl rather than
  // Program.

 private:
  CompileConfig compile_config_;

  uint64 ndarray_writer_counter_{0};
  uint64 ndarray_reader_counter_{0};
  int global_id_counter_{0};

  // SNode information that requires using Program.
  SNodeFieldMap snode_to_fields_;
  SNodeRwAccessorsBank snode_rw_accessors_bank_;

  std::vector<std::unique_ptr<SNodeTree>> snode_trees_;
  // Lazy cache for `get_snode_by_id`. Invalidated by `add_snode_tree` and `destroy_snode_tree`.
  std::unordered_map<int, SNode *> snode_id_cache_;
  // Adstack caching state (per-task metadata, bytecode, size-expr results, generation counters). All adstack-specific
  // surface lives in `program/adstack_size_expr_eval.{h,cpp}`; routed through `adstack_cache()` getter.
  std::unique_ptr<AdStackCache> adstack_cache_;

  // Adstack-sizing-info identity registry. See `register_adstack_sizing_info`. Index 0 is reserved as the
  // "no overflow" sentinel so the codegen-emitted `cmpxchg(0, id)` cleanly distinguishes "task id recorded"
  // from "slot still clean". Reverse lookup map keyed by `ad_stack_ptr` keeps `register_adstack_sizing_info`
  // idempotent across re-launches of the same kernel.
  std::vector<AdStackSizingInfoEntry> adstack_sizing_info_registry_{AdStackSizingInfoEntry{}};
  std::unordered_map<const void *, uint32_t> adstack_sizing_info_id_by_ptr_;
  mutable std::mutex adstack_sizing_info_registry_mutex_;
  std::stack<int> free_snode_tree_ids_;

  std::vector<std::unique_ptr<Function>> functions_;
  std::unordered_map<FunctionKey, Function *> function_map_;

  std::unique_ptr<ProgramImpl> program_impl_;
  float64 total_compilation_time_{0.0};
  static std::atomic<int> num_instances_;
  bool finalized_{false};
  size_t num_offloaded_tasks_on_last_call_{0};

  // TODO: Move ndarrays_ to be managed by runtime
  std::unordered_map<void *, std::unique_ptr<Ndarray>> ndarrays_;
};

}  // namespace quadrants::lang
