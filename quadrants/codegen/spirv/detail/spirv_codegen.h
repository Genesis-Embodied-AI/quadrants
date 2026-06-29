// In order to declare spirv::SType, spirv::Value, spirv::Label,
// we need to import #include <spirv/unified1/spirv.hpp>, which
// adds additional dependencies of spirv_codegen.h clients, so
// putting the class declaration in this separate file instead
// (we could also simply put the declaration in the .cpp file, but
// maybe using a separate header file is more standard?)

#pragma once

#include "quadrants/util/lang_util.h"

#include "quadrants/codegen/spirv/snode_struct_compiler.h"
#include "quadrants/codegen/spirv/kernel_utils.h"
#include "quadrants/codegen/spirv/spirv_ir_builder.h"
#include "quadrants/codegen/spirv/kernel_utils.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

#include <unordered_set>

namespace quadrants::lang {
struct CompileConfig;
namespace spirv {
namespace detail {

using BufferInfo = TaskAttributes::BufferInfo;
using BufferBind = TaskAttributes::BufferBind;
using BufferInfoHasher = TaskAttributes::BufferInfoHasher;

class TaskCodegen : public IRVisitor {
 public:
  struct Params {
    OffloadedStmt *task_ir;
    Arch arch;
    DeviceCapabilityConfig *caps;
    std::vector<CompiledSNodeStructs> compiled_structs;
    const KernelContextAttributes *ctx_attribs;
    std::string qd_kernel_name;
    int task_id_in_kernel;
    const CompileConfig *compile_config{nullptr};
  };

  const bool use_64bit_pointers = false;

  explicit TaskCodegen(const Params &params);

  void fill_snode_to_root();

  // Replace the wild '%' in the format string with "%%".
  std::string sanitize_format_string(std::string const &str);

  struct Result {
    std::vector<uint32_t> spirv_code;
    TaskAttributes task_attribs;
    std::unordered_map<std::vector<int>, irpass::ExternalPtrAccess, hashing::Hasher<std::vector<int>>> arr_access;
    // Access bits restricted to references through the `.grad` slot of each ndarray arg. Used by
    // `GfxRuntime::launch_kernel` to skip the host->device grad blit when no task in the kernel touches the
    // grad slot - the typical forward-pass kernel of reverse-mode AD, which reads / writes the primal data
    // slot only and leaves `.grad` alone until the backward dispatch.
    std::unordered_map<std::vector<int>, irpass::ExternalPtrAccess, hashing::Hasher<std::vector<int>>> grad_arr_access;
  };

  Result run();

  void visit(OffloadedStmt *) override;
  void visit(Block *stmt) override;
  void visit(PrintStmt *stmt) override;
  void visit(ConstStmt *const_stmt) override;
  void visit(AllocaStmt *alloca) override;
  void visit(MatrixPtrStmt *stmt) override;
  void visit(LocalLoadStmt *stmt) override;
  void visit(LocalStoreStmt *stmt) override;
  void visit(GetRootStmt *stmt) override;
  void visit(GetChStmt *stmt) override;
  void visit(SNodeOpStmt *stmt) override;
  void visit(SNodeLookupStmt *stmt) override;
  void visit(RandStmt *stmt) override;
  void visit(LinearizeStmt *stmt) override;
  void visit(LoopIndexStmt *stmt) override;
  void visit(GlobalStoreStmt *stmt) override;
  void visit(GlobalLoadStmt *stmt) override;
  void visit(ArgLoadStmt *stmt) override;
  void visit(GetElementStmt *stmt) override;
  void visit(ReturnStmt *stmt) override;
  void visit(GlobalTemporaryStmt *stmt) override;
  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override;
  void visit(ExternalPtrStmt *stmt) override;
  void visit(DecorationStmt *stmt) override;
  void visit(UnaryOpStmt *stmt) override;
  void visit(BinaryOpStmt *bin) override;
  void visit(TernaryOpStmt *tri) override;
  void visit(InternalFuncStmt *stmt) override;
  void visit(AtomicOpStmt *stmt) override;
  void visit(IfStmt *if_stmt) override;
  void visit(RangeForStmt *for_stmt) override;
  void visit(WhileStmt *stmt) override;
  void visit(WhileControlStmt *stmt) override;
  void visit(ContinueStmt *stmt) override;
  void visit(AdStackAllocaStmt *stmt) override;
  void visit(AdStackPushStmt *stmt) override;
  void visit(AdStackPopStmt *stmt) override;
  void visit(AdStackLoadTopStmt *stmt) override;
  void visit(AdStackLoadTopAdjStmt *stmt) override;
  void visit(AdStackAccAdjointStmt *stmt) override;

 private:
  void emit_headers();
  void generate_serial_kernel(OffloadedStmt *stmt);
  void gen_array_range(Stmt *stmt);
  void generate_range_for_kernel(OffloadedStmt *stmt);
  void generate_struct_for_kernel(OffloadedStmt *stmt);
  spirv::Value at_buffer(const Stmt *ptr, DataType dt);
  spirv::Value load_buffer(const Stmt *ptr, DataType dt, bool is_volatile = false);
  void store_buffer(const Stmt *ptr, spirv::Value val);
  spirv::Value get_buffer_value(BufferInfo buffer, DataType dt);
  spirv::Value make_pointer(size_t offset);
  void compile_args_struct();
  void compile_ret_struct();
  std::vector<BufferBind> get_buffer_binds();
  void push_loop_control_labels(spirv::Label continue_label, spirv::Label merge_label);
  void pop_loop_control_labels();
  const spirv::Label current_continue_label() const;
  const spirv::Label current_merge_label() const;
  const spirv::Label return_label() const;

  enum class ActivationOp { activate, deactivate, query };
  spirv::Value bitmasked_activation(ActivationOp op,
                                    spirv::Value parent_ptr,
                                    int root_id,
                                    const SNode *sn,
                                    spirv::Value input_index);
  void generate_overflow_branch(const spirv::Value &cond_v, const std::string &op, const std::string &tb);
  spirv::Value generate_uadd_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb);
  spirv::Value generate_usub_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb);
  spirv::Value generate_sadd_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb);
  spirv::Value generate_ssub_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb);
  spirv::Value generate_umul_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb);
  spirv::Value generate_smul_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb);
  spirv::Value generate_ushl_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb);
  spirv::Value generate_sshl_overflow(const spirv::Value &a, const spirv::Value &b, const std::string &tb);
  inline bool ends_with(std::string const &value, std::string const &ending);

  Arch arch_;
  DeviceCapabilityConfig *caps_;
  const CompileConfig *compile_config_;
  // Index of this task within its kernel's task list (`KernelCodegen::run` -> `tasks[i]` for offload-stmt `i`). Stored
  // from `Params::task_id_in_kernel` at construction so the LCA-block row-claim can OpAtomicIAdd into its own slot of
  // the per-kernel `BufferType::AdStackRowCounter` array. Per-task slots are what makes the post-launch host readback
  // usable - a single shared slot 0 would have the next task's bind clear it before the host reads, losing every task
  // except the last.
  int task_id_in_kernel_{0};

  struct BufferInfoTypeTupleHasher {
    std::size_t operator()(const std::pair<BufferInfo, int> &buf) const {
      return BufferInfoHasher()(buf.first) ^ (buf.second << 5);
    }
  };

  spirv::SType args_struct_type_;
  spirv::Value args_buffer_value_;

  std::unordered_map<std::vector<int>, spirv::SType, hashing::Hasher<std::vector<int>>> args_struct_types_;

  std::vector<spirv::SType> rets_struct_types_;

  spirv::SType ret_struct_type_;
  spirv::Value ret_buffer_value_;

  std::shared_ptr<spirv::IRBuilder> ir_;  // spirv binary code builder
  std::unordered_map<std::pair<BufferInfo, int>, spirv::Value, BufferInfoTypeTupleHasher> buffer_value_map_;
  std::unordered_map<std::pair<BufferInfo, int>, uint32_t, BufferInfoTypeTupleHasher> buffer_binding_map_;
  // All existing type views of each underlying storage buffer, in creation order. When a second or later view is minted
  // in `get_buffer_value`, we decorate every entry here with `Aliased` so the driver is forbidden from assuming the
  // views don't alias - otherwise a plain load through one view is not ordered against an atomic through another view
  // of the same memory, silently zeroing gradients on the load-and-clear reverse-mode pattern. See `get_buffer_value`
  // for the decoration site and the commit message for the full failure matrix.
  std::unordered_map<BufferInfo, std::vector<spirv::Value>, BufferInfoHasher> buffer_views_by_buffer_;
  std::unordered_set<uint32_t> aliased_decorated_buffer_ids_;
  std::vector<spirv::Value> shared_array_binds_;
  spirv::Value kernel_function_;
  spirv::Label kernel_return_label_;
  bool gen_label_{false};

  int binding_head_{2};  // Args:0, Ret:1

  OffloadedStmt *const task_ir_;  // not owned
  std::vector<CompiledSNodeStructs> compiled_structs_;
  std::unordered_map<int, int> snode_to_root_;
  const KernelContextAttributes *const ctx_attribs_;  // not owned
  const std::string task_name_;
  std::vector<spirv::Label> continue_label_stack_;
  std::vector<spirv::Label> merge_label_stack_;

  std::unordered_set<const Stmt *> offload_loop_motion_;

  TaskAttributes task_attribs_;
  std::unordered_map<int, GetRootStmt *> root_stmts_;  // maps root id to get root stmt
  std::unordered_map<const Stmt *, BufferInfo> ptr_to_buffers_;
  // Shared float AllocaStmts targeted by atomics, populated by
  // scan_shared_atomic_allocs() before codegen. Value = true means the alloca
  // has non-add ops (CAS unconditionally needed); false = add-only (native
  // shared float atomics can be used if the device supports them).
  std::unordered_map<const Stmt *, bool> shared_float_allocas_with_atomic_rmw_;
  // Propagated from shared_float_allocas_with_atomic_rmw_ to derived
  // MatrixPtrStmt nodes during codegen, so that load/store/atomic visitors
  // know to bitcast. E.g. if `sharr` (AllocaStmt) is retyped, then
  // `sharr[0]` (MatrixPtrStmt) is added here during visit(MatrixPtrStmt).
  std::unordered_set<const Stmt *> uint_backed_shared_float_ptr_stmts_;
  std::unordered_map<std::vector<int>, Value, hashing::Hasher<std::vector<int>>> argid_to_tex_value_;

  struct PhysicalPtrComponents {
    spirv::Value base_ptr;
    spirv::Value element_index;
  };
  std::unordered_map<const Stmt *, PhysicalPtrComponents> physical_ptr_components_;

  bool use_volatile_buffer_access_{false};

  // Where the primal/adjoint storage for an AdStack lives. `heap_float` backs f32 adstacks and `heap_int` backs i32 and
  // u1 adstacks (u1 stored as i32 to match `get_array_type`'s bool->int remap on the Function-scope path); other
  // primitive types are hard-errored by `visit(AdStackAllocaStmt)`, so no Function-scope fallback exists. Each kind
  // maps to its own per-dispatch StorageBuffer (`BufferType::AdStackHeapFloat` / `BufferType::AdStackHeapInt`).
  enum class AdStackHeapKind { heap_float, heap_int };
  struct AdStackSpirv {
    spirv::Value count_var;  // u32, Function scope - current number of entries
    AdStackHeapKind heap_kind;
    // Index of this alloca in the task's pre-scan order; also the shader-side slot index into the
    // `AdStackMetadata` buffer (entries are `stride_float, stride_int, (offset_i, max_size_i)*`).
    uint32_t stack_id{0};
    // Compile-time bound carried alongside the runtime-metadata index so the host launcher can
    // populate the metadata buffer from `max_size_compile_time` when the per-alloca `size_expr` is
    // empty (offline-cache load). Never read by the shader itself.
    uint32_t max_size_compile_time{0};
    // Compile-time prefix-sum offset (in elements of the heap's element type). Mirrored into
    // `TaskAttributes::AdStackSizingAttribs::allocas[stack_id].offset_in_elems_compile_time` so the
    // host launcher's no-size_expr path publishes the same layout the codegen assumed.
    uint32_t offset_in_elems_compile_time{0};
    spirv::SType elem_type;
    // Per-alloca cached loads from the AdStackMetadata buffer (offset, max_size, and the derived
    // `adjoint_offset = offset + max_size`). Lazily emitted on the first push/load_top/load_top_adj
    // visitor for this alloca, reused for every subsequent site.
    spirv::Value offset_val;
    spirv::Value max_size_val;
    spirv::Value adjoint_offset_val;
  };
  std::unordered_map<const Stmt *, AdStackSpirv> ad_stacks_;
  // Total per-thread heap strides, pre-computed from the IR before any visitor runs from the
  // compile-time `max_size` on each alloca. The runtime recomputes these from the evaluated
  // `size_expr` trees and publishes them into the `AdStackMetadata` buffer; the shader reads the
  // per-dispatch values from the metadata buffer rather than using these immediates directly. The
  // compile-time values are still mirrored into `task_attribs.ad_stack.*` so the offline-cache-hit
  // path with no symbolic bound captured reproduces the pre-PR shader layout.
  uint32_t ad_stack_heap_per_thread_stride_float_{0};
  uint32_t ad_stack_heap_per_thread_stride_int_{0};
  // Total number of `AdStackAllocaStmt` the body will visit, pre-computed by the same scan that builds the heap
  // strides. Used to size a single shared Function-scope `uint[num_ad_stacks_]` count array (see
  // `ad_stack_count_ptr`) so that all per-stack `count` values share one OpVariable accessed via
  // OpAccessChain. The shared array prevents spirv-opt's `LocalMultiStoreElim` / `SSARewrite` from promoting each
  // count to a separate phi at every enclosing loop header - reverse-grad kernels that allocate hundreds of
  // adstacks across enclosing loops would otherwise grow phi mega-clusters of hundreds of entries per loop
  // header, which spirv-cross emits as one `uint _N;` forward-decl and one `_N = _N;` alias copy per phi per
  // predecessor branch in the cross-compiled MSL.
  uint32_t num_ad_stacks_{0};
  // Single Function-scope `uint[num_ad_stacks_]` array, allocated lazily on first `ad_stack_count_ptr` call.
  // `info.count_var` for each stack is now an `OpAccessChain` pointer into element `stack_id` of this array
  // rather than its own `OpVariable Function`.
  spirv::Value ad_stack_count_array_var_;
  // Running offsets into the per-thread slice assigned to the next AdStackAllocaStmt visitor. Each ends equal to
  // the corresponding stride once every alloca has been visited; these feed the
  // `offset_in_elems_compile_time` of each alloca's `AdStackSizingAttribs::allocas` entry.
  uint32_t ad_stack_heap_next_offset_float_{0};
  uint32_t ad_stack_heap_next_offset_int_{0};
  // Buffers are cached for reuse across push/pop/load-top visitors and (re)computed lazily on first use inside a
  // task so the `OpLoad` falls inside the dispatch body rather than the function header.
  spirv::Value ad_stack_heap_buffer_float_;
  spirv::Value ad_stack_heap_buffer_int_;
  // No SSA cache for the per-thread heap base: the heap base is `row_id_var * stride`, where `row_id_var` is a
  // Function-scope OpVariable load. Per-call-site OpLoad yields a fresh SSA in the call site's basic block, so a single
  // cached SSA cannot be reused across sibling blocks of the LCA without violating SPIR-V section 2.16 dominance.
  // `get_ad_stack_heap_thread_base_float()` / `_int()` therefore re-emit the load + multiply at every push / load-top /
  // load-top-adj. spirv-opt and spirv-cross still CSE redundant loads inside a single basic block, so the only added
  // cost is one OpIMul per push site that lives in a different block. Cached handle to the AdStackMetadata
  // StorageBuffer and the per-task stride values loaded from its header slots. Same dominance rule as the heap thread
  // bases - eager emission at the first alloca site of its heap kind, reused at every downstream
  // push/load-top/load-top-adj.
  spirv::Value ad_stack_metadata_buffer_;
  spirv::Value ad_stack_metadata_stride_float_;
  spirv::Value ad_stack_metadata_stride_int_;
  // Lowest common dominator (LCA) block of every f32-typed AdStackPushStmt / AdStackLoadTopStmt / AdStackLoadTopAdjStmt
  // in the task body, populated by the pre-pass scan in `run()` that also builds the heap strides. The LCA is where
  // `visit(Block *)` emits the one-shot row-claim that materialises `ad_stack_row_id_var_float_`. Computed only over
  // float-typed pushes deliberately: int-heap pushes for loop index recovery and if-branch flags often live
  // unconditionally at the offload body root (the autodiff pass emits them outside any user gate so the reverse pass
  // can replay control flow), and folding them into the LCA computation pulls the LCA up to the root for kernels with
  // grid-style sparse predicates - eliminating the savings on the float heap, which is the only one large enough to
  // matter (per-thread float strides measured in thousands of f32 elements dominate the footprint, while int-stack
  // strides are typically two orders of magnitude smaller). `nullptr` when the task has no f32 adstack pushes (the
  // float heap is unbound and no row-claim is emitted) or when the LCA reduces to the task body's root - in the latter
  // case the claim still runs from the root, equivalent in row-occupancy to the prior `invoc_id`-keyed eager layout.
  Block *ad_stack_lca_block_float_{nullptr};
  // Set of `AdStackPushStmt`s recognized as autodiff-bootstrap const-init pushes by the LCA pre-pass: parent block is
  // the offload body, previous sibling is the matching alloca, pushed value is a `ConstStmt`. These pushes run
  // unconditionally on every dispatched thread, so the LCA computation skips them (folding their parent block in would
  // drag the LCA up to the offload root and revert to per-thread sizing); the `visit(AdStackPushStmt)` visitor also
  // skips the slot store for these (the matching reverse pop only decrements `count_var` and never reads the slot back
  // via `load_top`, so the bootstrap value is dead memory and writing it through a possibly-unclaimed `row_id_var`
  // would corrupt arbitrary heap rows). Only the `count_var` increment is kept so push and pop stay balanced.
  std::unordered_set<AdStackPushStmt *> ad_stack_bootstrap_pushes_;

  // Function-scope u32 alloca holding the maximum-overflow-signal-seen-this-thread across every adstack push site
  // in the task. Each `AdStackPushStmt` visitor updates it with `signal = (count >= max) ? stack_id + 1 : 0` via
  // OpUMax (register-only, no atomic, no global memory access on the not-overflowing fast path). Right before the
  // kernel's `OpReturn`, an `if any_overflow_signal_var > 0 then atomic_fetch_max(host_visible_buf[0],
  // any_overflow_signal_var)` emits the host-visible signal exactly once per task per kernel. The host polls the
  // buffer (Apple Silicon: shared memory; Vulkan: HOST_VISIBLE | HOST_COHERENT) without any DtoH or sync drain.
  // Lazy-allocated by the first `AdStackPushStmt` visitor; nullptr when the task has no adstack pushes (no
  // task-end emit needed). Reset to {} per task in `generate_*_kernel`.
  spirv::Value any_overflow_signal_var_;
  // Set to true when the task body contains at least one `AdStackPushStmt` visit. Read by
  // `emit_adstack_task_end_overflow_check` to decide whether to emit the task-end overflow check and
  // its AdStackOverflow / AdStackTaskRegistryId buffer accesses. Forward-only tasks never set this and
  // therefore never request the AdStack-side buffer binds, which keeps the SPIR-V launcher's bind
  // path from null-binding `AdStackTaskRegistryId` on a Program that has not allocated it
  // (allocation only fires inside `publish_adstack_metadata_spirv` for kernels with at least one
  // adstack-bearing task; null-binding would crash Metal's `rw_buffer` device-equality assertion).
  bool task_has_adstack_push_{false};
  // Function-scope OpVariable<u32> initialized to UINT32_MAX at task entry; overwritten with the atomically claimed row
  // index when codegen visits `ad_stack_lca_block_float_`. `get_ad_stack_heap_thread_base_float()` loads this variable
  // and multiplies against the runtime float stride to produce the per-thread heap base, replacing the prior `invoc_id
  // * stride` formula. The variable is per-invocation (Function storage class) so the load yields a fresh SSA at each
  // push site without violating SPIR-V section 2.16 dominance even when push sites live in sibling blocks of the LCA.
  // The int heap path uses the eager `gl_GlobalInvocationID * stride_int` layout in
  // `get_ad_stack_heap_thread_base_int()` and does not consult any row_id_var.
  spirv::Value ad_stack_row_id_var_float_;
  // Cached SSA handle to the per-dispatch StorageBuffer holding the single u32 atomic counter
  // (`BufferType::AdStackRowCounter`). Lazily populated on first use inside the LCA-block claim emission so the
  // `OpAtomicIAdd` lives in the dispatch body rather than the function header. Zero (default-constructed) when the task
  // has no adstack push sites and the buffer is not bound.
  spirv::Value ad_stack_row_counter_buffer_;
  // Cached SSA handle to the per-kernel `BufferType::AdStackBoundRowCapacity` (`uint[num_tasks_in_kernel]`). Lazily
  // populated at the float Lowest Common Ancestor (LCA) block emission site when the defense-in-depth bounds check
  // fires; the host writes the per-task capacity (the reducer's count for tasks with a captured `bound_expr`,
  // UINT32_MAX otherwise) so the OpAtomicUMax sentinel only fires on a reducer / main divergence. Zero-default when the
  // task has no float adstack push sites and the buffer is not bound.
  spirv::Value ad_stack_bound_row_capacity_buffer_;
  // Return (lazily) the StorageBuffer of `Array<f32>` that backs f32 adstacks for this dispatch, and the per-thread
  // base index inside it.
  spirv::Value get_ad_stack_heap_buffer_float();
  spirv::Value get_ad_stack_heap_thread_base_float();
  spirv::Value ad_stack_heap_float_ptr(spirv::Value slot_offset, spirv::Value count);
  // Same accessors for the int-typed heap buffer (backs i32 and u1 adstacks).
  spirv::Value get_ad_stack_heap_buffer_int();
  spirv::Value get_ad_stack_heap_thread_base_int();
  spirv::Value ad_stack_heap_int_ptr(spirv::Value slot_offset, spirv::Value count);
  // Metadata buffer accessors. Each emits one OpLoad on first use and caches the SSA id.
  spirv::Value get_ad_stack_metadata_buffer();
  spirv::Value get_ad_stack_metadata_stride_float();
  spirv::Value get_ad_stack_metadata_stride_int();
  void ensure_ad_stack_metadata_loaded(AdStackSpirv &info);
  // Lazy-allocate the per-task overflow-signal accumulator at the function entry block. See the member
  // doc on `any_overflow_signal_var_` for the design rationale.
  spirv::Value ensure_any_overflow_signal_var();
  // Emit the task-end overflow check at the current insertion point. Must be called from each
  // `generate_*_kernel` just before the closing `OpReturn`. No-op when the task has no adstack push sites.
  void emit_adstack_task_end_overflow_check();
  // Routes to the correct backing-typed pointer (`*f32` for `heap_float`, `*i32` for `heap_int`) based on
  // `info.heap_kind`. See comment on the implementation for the bool<->i32 conversion contract.
  spirv::Value ad_stack_slot_ptr(AdStackSpirv &info, spirv::Value idx, bool primal);
  spirv::SType ad_stack_backing_type(const AdStackSpirv &info) const;
  // OpAccessChain pointer to element `stack_id` of the shared Function-scope count array. Lazily allocates
  // `ad_stack_count_array_var_` on first call (sized at `num_ad_stacks_` from the pre-pass scan). Returned
  // pointer feeds `info.count_var` and works with the existing `load_variable` / `store_variable` helpers.
  spirv::Value ad_stack_count_ptr(uint32_t stack_id);
};
}  // namespace detail
}  // namespace spirv
}  // namespace quadrants::lang
