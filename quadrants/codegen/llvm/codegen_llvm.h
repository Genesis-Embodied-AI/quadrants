// The LLVM backend for CPUs/NVPTX/AMDGPU
#pragma once

#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>

#ifdef QD_WITH_LLVM

#include "quadrants/ir/ir.h"
#include "quadrants/codegen/llvm/llvm_codegen_utils.h"
#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/program/program.h"

namespace quadrants::lang {

class TaskCodeGenLLVM;

class FunctionCreationGuard {
 public:
  TaskCodeGenLLVM *mb;
  llvm::Function *old_func;
  llvm::Function *body;
  llvm::BasicBlock *old_entry, *allocas, *entry, *old_final, *final;
  llvm::IRBuilder<>::InsertPoint ip;

  FunctionCreationGuard(TaskCodeGenLLVM *mb, std::vector<llvm::Type *> arguments, const std::string &func_name);

  ~FunctionCreationGuard();
};

class TaskCodeGenLLVM : public IRVisitor, public LLVMModuleBuilder {
 public:
  const CompileConfig &compile_config;
  const Kernel *kernel;
  IRNode *ir;
  Program *prog;
  std::string kernel_name;
  std::vector<llvm::Value *> kernel_args;
  llvm::Type *context_ty;
  llvm::Type *physical_coordinate_ty;
  llvm::Value *current_coordinates;
  llvm::Value *parent_coordinates{nullptr};
  llvm::Value *block_corner_coordinates{nullptr};
  llvm::GlobalVariable *bls_buffer{nullptr};
  // Mainly for supporting continue stmt
  llvm::BasicBlock *current_loop_reentry;
  // Mainly for supporting break stmt
  llvm::BasicBlock *current_while_after_loop;
  llvm::FunctionType *task_function_type;
  std::unordered_map<Stmt *, llvm::Value *> llvm_val;
  llvm::Function *func;
  OffloadedStmt *current_offload{nullptr};
  std::unique_ptr<OffloadedTask> current_task;
  std::vector<OffloadedTask> offloaded_tasks;
  llvm::BasicBlock *func_body_bb;
  llvm::BasicBlock *final_block;
  std::set<std::string> linked_modules;
  bool returned{false};
  std::unordered_set<int> used_tree_ids;
  std::unordered_set<int> struct_for_tls_sizes;
  const Callable *current_callable{nullptr};

  // The task_codegen_id represents the id of the offloaded task
  int task_codegen_id{0};

  // Per-task heap-backed adstack state. Replaces the function-scope `create_entry_block_alloca` that used to
  // bound the cumulative adstack size by the worker-thread stack limit (~512 KB on macOS secondary threads).
  // `ad_stack_per_thread_stride_` is the sum of `AdStackAllocaStmt::size_in_bytes()` (aligned up to 8) for every
  // adstack in the current offloaded task. `ad_stack_offsets_` is indexed by each alloca's `stack_id` (assigned
  // during the pre-scan in declaration order) and stores the offset within the per-thread slice (i.e. the sum of
  // sizes of siblings visited earlier in the pre-scan). Both are populated by a pre-scan of the task body in
  // `init_offloaded_task_function` before any codegen runs, so later sibling allocas do not shift an earlier
  // alloca's offset out from under a cached SSA pointer. The split-heap helpers below cache the per-kind base SSA
  // values; tasks address through `_float` / `_int` exclusively.
  std::size_t ad_stack_per_thread_stride_{0};
  // Per-thread strides per heap kind. Float allocas live on the lazy float heap (sized by the launcher to the count of
  // threads passing the captured `bound_expr` gate, when one is recognized); int allocas live on the eager int heap
  // (sized to `num_threads * stride_int`). Each alloca's `ad_stack_offsets_[stack_id]` is the byte offset within its
  // slice of the appropriate kind, NOT within a combined slice.
  std::size_t ad_stack_per_thread_stride_float_{0};
  std::size_t ad_stack_per_thread_stride_int_{0};
  std::vector<std::size_t> ad_stack_offsets_;
  // Mirror of the pre-scan output copied into `current_task->ad_stack` in `finalize_offloaded_task_function`. Kept
  // as class state so the scan (which runs before `current_task` is constructed) can still push entries in order.
  std::vector<AdStackAllocaInfo> ad_stack_allocas_info_;
  std::vector<SerializedSizeExpr> ad_stack_size_exprs_;
  // Cached SSA bases for the split float / int heaps, loaded once at the top of the task body via
  // `LLVMRuntime_get_adstack_heap_buffer_float` / `_int` and reused at every per-alloca base computation.
  llvm::Value *ad_stack_heap_base_float_llvm_{nullptr};
  llvm::Value *ad_stack_heap_base_int_llvm_{nullptr};
  // Cached SSA values for the per-launch metadata fields the host publishes into
  // `LLVMRuntime.adstack_{per_thread_stride_float,per_thread_stride_int,offsets,max_sizes}` before each dispatch.
  // Loaded once at `entry_block` (via `ensure_ad_stack_metadata_llvm`) and reused by every `AdStack*` visit. Resolving
  // via runtime fields lets `AdStackAllocaStmt`'s base-address math and `AdStackPushStmt`'s overflow bound scale per
  // launch from `SizeExpr` without a recompile. `ad_stack_stride_llvm_` is the legacy combined stride loaded from the
  // deprecated `LLVMRuntime_get_adstack_per_thread_stride` getter; new code paths read the split fields below directly.
  llvm::Value *ad_stack_stride_llvm_{nullptr};
  llvm::Value *ad_stack_stride_float_llvm_{nullptr};
  llvm::Value *ad_stack_stride_int_llvm_{nullptr};
  llvm::Value *ad_stack_offsets_ptr_llvm_{nullptr};
  llvm::Value *ad_stack_max_sizes_ptr_llvm_{nullptr};
  // Float-heap lazy claim state. `ad_stack_lca_block_float_ir_` is the IR-level Block at which the codegen emits the
  // one-shot atomic-rmw row claim into `LLVMRuntime.adstack_row_counters[task_id]`; the LLVM-side claim emit uses the
  // current builder insertion point at the matching IR-block visit, so no separate LLVM-block cache is needed.
  // `ad_stack_row_id_var_float_llvm_` is a Function-scope `alloca i32` initialised to UINT32_MAX at task entry; the
  // claim site writes the atomic-add result, and every per-alloca base computation for a float-typed alloca reads it
  // back. Threads that never reach the LCA never claim a row and never touch the float heap, which is exactly the
  // property the captured `bound_expr` reducer relies on to size the heap.
  Block *ad_stack_lca_block_float_ir_{nullptr};
  llvm::Value *ad_stack_row_id_var_float_llvm_{nullptr};
  // Set of autodiff-bootstrap const-init pushes identified by the shared analysis: `push(stack, ConstStmt)` whose
  // parent block is the offload body and whose previous sibling is the matching alloca. The `visit(AdStackPushStmt)`
  // visitor skips the slot store at these sites (only the count_var increment is kept so push and pop stay balanced),
  // because the bootstrap value is dead memory (no `load_top` ever reads it back) and writing through a
  // possibly-unclaimed `row_id_var` would corrupt arbitrary heap rows.
  std::unordered_set<AdStackPushStmt *> ad_stack_bootstrap_pushes_;
  // Set of f32-typed `AdStackAllocaStmt`s the codegen must address lazily through the split float heap (because the
  // task captured a `bound_expr`). The base for these allocas changes after the LCA-block atomic-rmw claim updates
  // `ad_stack_row_id_var_float_llvm_`, so `visit(AdStackAllocaStmt)` does not cache a static base in `llvm_val[stmt]`;
  // every push / load-top / load-top-adj / pop site calls `get_ad_stack_base_llvm(stack)` which computes `heap_float +
  // row_id_var * stride_float + offset` at the call site. Int / u1 allocas in the same task use the eager split-int
  // layout (`heap_int + linear_tid * stride_int + offset`); both paths skip the legacy combined-heap addressing.
  std::unordered_set<AdStackAllocaStmt *> ad_stack_lazy_float_allocas_;
  // Helpers that load the split-heap runtime fields once at `entry_block`. `ensure_ad_stack_heap_base_split_llvm`
  // caches the float / int heap base pointers; `ensure_ad_stack_metadata_split_llvm` adds the per-kind strides on top
  // of the legacy combined stride / offsets / max_sizes loads. Tasks without a captured `bound_expr` keep the
  // combined-heap path and never call into these.
  void ensure_ad_stack_heap_base_split_llvm();
  void ensure_ad_stack_metadata_split_llvm();
  // Returns (creating on first call) the Function-scope `alloca i32` initialised to UINT32_MAX at task entry that holds
  // this thread's lazily-claimed float-heap row id. The atomic-rmw claim at the float LCA block overwrites it with the
  // value the launcher's row counter returns; downstream float push / load-top sites read it back to compute their
  // per-thread base. Threads that never reach the LCA never claim a row and never touch the float heap.
  llvm::Value *ensure_ad_stack_row_id_var_float_llvm();
  // Emit the float-heap lazy row claim at the current insertion point. Called from `visit(Block *)` exactly once per
  // task at the IR-level Lowest Common Ancestor (LCA) of every f32 push / load-top site. Atomic-adds 1 into
  // `runtime->adstack_row_counters[task_codegen_id]`, clamps against `runtime->adstack_bound_row_capacities[task_
  // codegen_id]`, stores the result into `ad_stack_row_id_var_float_llvm_`. Threads that never reach this block never
  // claim a row.
  void emit_ad_stack_row_claim_llvm();
  // Return the per-thread base pointer for `stack`. For lazy float allocas (in tasks with `bound_expr`), emits
  // `heap_float + row_id_var * stride_float + offset` at the current insertion point - because `row_id_var` changes
  // after the LCA-block atomic-rmw, the base must be recomputed at every push / load-top / load-top-adj / pop site
  // rather than cached in `llvm_val[stack]`. For all other allocas (eager int in split-layout tasks and any alloca in
  // combined-layout tasks), returns the cached `llvm_val[stack]` set by `visit(AdStackAllocaStmt)`.
  llvm::Value *get_ad_stack_base_llvm(AdStackAllocaStmt *stack);
  // Captured static gate predicate from the shared analysis. Propagated through to `current_task->ad_stack.bound_expr`
  // so the host launcher can dispatch the per-arch reducer to size the float heap to the actual gate-passing thread
  // count.
  std::optional<StaticAdStackBoundExpr> ad_stack_static_bound_expr_;
  // Per-task per-stack `alloca i64` holding the live push count, hoisted to the entry block so `mem2reg` can promote it
  // to SSA and `GVN` can fold consecutive count loads / stores across straight-line unrolled bodies. Replaces the
  // heap-resident `u64` count header at `stack_ptr[0..8)` for every AdStack op when `compile_config.debug == false`.
  // The 8-byte heap header gap is preserved for layout compatibility but is never read or written from kernel code on
  // the release path.
  std::unordered_map<const AdStackAllocaStmt *, llvm::Value *> ad_stack_count_alloca_llvm_;

  std::unordered_map<const Stmt *, std::vector<llvm::Value *>> loop_vars_llvm;

  std::unordered_map<Function *, llvm::Function *> func_map;

  using IRVisitor::visit;
  using LLVMModuleBuilder::call;

  explicit TaskCodeGenLLVM(int id,
                           const CompileConfig &config,
                           QuadrantsLLVMContext &tlctx,
                           const Kernel *kernel,
                           IRNode *ir,
                           std::unique_ptr<llvm::Module> &&module = nullptr);

  Arch current_arch() const {
    return compile_config.arch;
  }

  void initialize_context();

  llvm::Value *get_arg(int i);

  llvm::Value *get_struct_arg(const std::vector<int> &index, bool create_load);

  llvm::Value *get_args_ptr(const Callable *callable, llvm::Value *context);

  void set_args_ptr(Callable *callable, llvm::Value *context, llvm::Value *ptr);

  llvm::Value *get_context();

  llvm::Value *get_tls_base_ptr();

  llvm::Type *get_tls_buffer_type();

  std::vector<llvm::Type *> get_xlogue_argument_types();

  std::vector<llvm::Type *> get_mesh_xlogue_argument_types();

  llvm::Type *get_xlogue_function_type();

  llvm::Type *get_mesh_xlogue_function_type();

  llvm::IntegerType *get_integer_type(int bits);

  llvm::Value *get_root(int snode_tree_id);

  llvm::Value *get_runtime();

  void emit_struct_meta_base(const std::string &name, llvm::Value *node_meta, SNode *snode);

  void create_elementwise_binary(BinaryOpStmt *stmt,
                                 std::function<llvm::Value *(llvm::Value *lhs, llvm::Value *rhs)> f);

  void create_elementwise_cast(UnaryOpStmt *stmt,
                               llvm::Type *to_ty,
                               std::function<llvm::Value *(llvm::Value *, llvm::Type *)> f,
                               bool on_self = false);

  std::unique_ptr<RuntimeObject> emit_struct_meta_object(SNode *snode);

  llvm::Value *emit_struct_meta(SNode *snode);

  virtual void emit_to_module();

  void eliminate_unused_functions();

  /**
   * @brief Runs the codegen and produces the compiled result.
   *
   * After this call, `module` and `tasks` will be moved.
   *
   * @return LLVMCompiledTask
   */
  virtual LLVMCompiledTask run_compilation();
  // For debugging only
  virtual llvm::Value *create_print(std::string tag, DataType dt, llvm::Value *value);

  llvm::Value *create_print(std::string tag, llvm::Value *value);

  void set_struct_to_buffer(const StructType *struct_type, llvm::Value *buffer, const std::vector<Stmt *> &elements);

  llvm::Value *cast_pointer(llvm::Value *val, std::string dest_ty_name, int addr_space = 0);

  void emit_list_gen(OffloadedStmt *listgen);

  void emit_gc(OffloadedStmt *stmt);

  llvm::Value *call(SNode *snode,
                    llvm::Value *node_ptr,
                    const std::string &method,
                    const std::vector<llvm::Value *> &arguments);

  llvm::Function *get_struct_function(const std::string &name, int tree_id);

  template <typename... Args>
  llvm::Value *call_struct_func(int tree_id, const std::string &func_name, Args &&...args);

  void create_increment(llvm::Value *ptr, llvm::Value *value);

  // Direct translation
  void create_naive_range_for(RangeForStmt *for_stmt);

  static std::string get_runtime_snode_name(SNode *snode);

  void visit(Block *stmt_list) override;

  void visit(AllocaStmt *stmt) override;

  void visit(RandStmt *stmt) override;

  virtual void emit_extra_unary(UnaryOpStmt *stmt);

  void visit(DecorationStmt *stmt) override;

  void visit(UnaryOpStmt *stmt) override;

  void visit(BinaryOpStmt *stmt) override;

  void visit(TernaryOpStmt *stmt) override;

  void visit(IfStmt *if_stmt) override;

  void visit(PrintStmt *stmt) override;

  void visit(ConstStmt *stmt) override;

  void visit(WhileControlStmt *stmt) override;

  void visit(ContinueStmt *stmt) override;

  void visit(WhileStmt *stmt) override;

  void visit(RangeForStmt *for_stmt) override;

  void visit(ArgLoadStmt *stmt) override;

  void visit(ReturnStmt *stmt) override;

  void visit(LocalLoadStmt *stmt) override;

  void visit(LocalStoreStmt *stmt) override;

  void visit(AssertStmt *stmt) override;

  void visit(SNodeOpStmt *stmt) override;

  llvm::Value *atomic_add_quant_fixed(llvm::Value *ptr,
                                      llvm::Type *physical_type,
                                      QuantFixedType *qfxt,
                                      llvm::Value *value);

  llvm::Value *atomic_add_quant_int(llvm::Value *ptr,
                                    llvm::Type *physical_type,
                                    QuantIntType *qit,
                                    llvm::Value *value,
                                    bool value_is_signed);

  llvm::Value *to_quant_fixed(llvm::Value *real, QuantFixedType *qfxt);

  virtual llvm::Value *optimized_reduction(AtomicOpStmt *stmt);

  virtual llvm::Value *quant_type_atomic(AtomicOpStmt *stmt);

  virtual llvm::Value *integral_type_atomic(AtomicOpStmt *stmt);

  virtual llvm::Value *atomic_op_using_cas(llvm::Value *output_address,
                                           llvm::Value *val,
                                           std::function<llvm::Value *(llvm::Value *, llvm::Value *)> op,
                                           const DataType &type);

  virtual llvm::Value *real_type_atomic(AtomicOpStmt *stmt);

  void visit(AtomicOpStmt *stmt) override;

  void visit(GlobalPtrStmt *stmt) override;

  void visit(MatrixPtrStmt *stmt) override;

  void store_quant_int(llvm::Value *ptr, llvm::Type *physical_type, QuantIntType *qit, llvm::Value *value, bool atomic);

  void store_quant_fixed(llvm::Value *ptr,
                         llvm::Type *physical_type,
                         QuantFixedType *qfxt,
                         llvm::Value *value,
                         bool atomic);

  void store_masked(llvm::Value *ptr, llvm::Type *ty, uint64 mask, llvm::Value *value, bool atomic);

  void visit(GlobalStoreStmt *stmt) override;

  llvm::Value *quant_int_or_quant_fixed_to_bits(llvm::Value *val, Type *input_type, llvm::Type *output_type);

  void visit(BitStructStoreStmt *stmt) override;

  void store_quant_floats_with_shared_exponents(BitStructStoreStmt *stmt);

  llvm::Value *extract_quant_float(llvm::Value *physical_value, BitStructType *bit_struct, int digits_id);

  llvm::Value *extract_quant_int(llvm::Value *physical_value, llvm::Value *bit_offset, QuantIntType *qit);

  llvm::Value *reconstruct_quant_fixed(llvm::Value *digits, QuantFixedType *qfxt);

  llvm::Value *reconstruct_quant_float(llvm::Value *input_digits,
                                       llvm::Value *input_exponent_val,
                                       QuantFloatType *qflt,
                                       bool shared_exponent);

  virtual llvm::Value *create_intrinsic_load(llvm::Value *ptr, llvm::Type *ty);

  void create_global_load(GlobalLoadStmt *stmt, bool should_cache_as_read_only);

  void visit(GlobalLoadStmt *stmt) override;

  void visit(GetRootStmt *stmt) override;

  void visit(LinearizeStmt *stmt) override;

  void visit(IntegerOffsetStmt *stmt) override;

  llvm::Value *create_bit_ptr(llvm::Value *byte_ptr, llvm::Value *bit_offset);

  std::tuple<llvm::Value *, llvm::Value *> load_bit_ptr(llvm::Value *bit_ptr);

  void visit(SNodeLookupStmt *stmt) override;

  void visit(GetChStmt *stmt) override;

  void visit(ExternalPtrStmt *stmt) override;

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override;

  void visit(ExternalTensorBasePtrStmt *stmt) override;

  virtual bool kernel_argument_by_val() const {
    return false;  // on CPU devices just pass in a pointer
  }

  std::string init_offloaded_task_function(OffloadedStmt *stmt, std::string suffix = "");

  // GPU-side `qd.checkpoint` gating prologue. Emits an LLVM-IR early-return at the top of `func_body_bb` (set up by
  // `init_offloaded_task_function`) that reads `RuntimeContext::checkpoint_resume_point_ptr` /
  // `checkpoint_yield_signal_ptr` and jumps to `final_block` whenever the checkpoint should be skipped.
  //
  // Used by the GPU backends as the gating mechanism on backends without conditional-graph
  // node support: pre-Hopper CUDA (no CUDA 12.4+ conditional nodes) and AMDGPU / HIP (no
  // HIP equivalent of conditional nodes or indirect dispatch in HIP 7.2). On CUDA SM 9.0+
  // the conditional-graph-node gate prevents the body kernel from launching when the
  // checkpoint should be skipped, so the prologue is dead code in the common path but kept
  // for correctness on overlapping-gate races. Not emitted by CPU codegen (the host launcher
  // does branch gating before launch) or by the GFX (Vulkan / Metal) codegen path (those use
  // indirect-dispatch gating via SPIR-V gate shaders that don't go through RuntimeContext).
  //
  // `cp_id` is the literal checkpoint id this task belongs to (-1 for non-checkpoint tasks,
  // in which case callers should not invoke this method).
  void emit_checkpoint_gate_prologue(int cp_id);

  void finalize_offloaded_task_function();

  FunctionCreationGuard get_function_creation_guard(std::vector<llvm::Type *> argument_types,
                                                    const std::string &func_name = "function_body");

  std::tuple<llvm::Value *, llvm::Value *> get_range_for_bounds(OffloadedStmt *stmt);

  virtual void create_offload_range_for(OffloadedStmt *stmt) = 0;

  virtual void create_offload_mesh_for(OffloadedStmt *stmt) {
    QD_NOT_IMPLEMENTED;
  }

  void create_offload_struct_for(OffloadedStmt *stmt);

  void visit(LoopIndexStmt *stmt) override;

  void visit(LoopLinearIndexStmt *stmt) override;

  void visit(BlockCornerIndexStmt *stmt) override;

  void visit(GlobalTemporaryStmt *stmt) override;

  void visit(ThreadLocalPtrStmt *stmt) override;

  void visit(BlockLocalPtrStmt *stmt) override;

  void visit(ClearListStmt *stmt) override;

  void visit(InternalFuncStmt *stmt) override;

  // Stack statements

  void ensure_ad_stack_metadata_llvm();
  llvm::Value *ensure_ad_stack_count_alloca_llvm(const AdStackAllocaStmt *stack);
  llvm::Value *emit_ad_stack_top_slot_ptr(const AdStackAllocaStmt *stack,
                                          llvm::Value *count,
                                          std::size_t adjoint_offset_bytes);
  llvm::Value *emit_ad_stack_single_slot_ptr(const AdStackAllocaStmt *stack, std::size_t adjoint_offset_bytes);

  void visit(AdStackAllocaStmt *stmt) override;

  void visit(AdStackPopStmt *stmt) override;

  void visit(AdStackPushStmt *stmt) override;

  void visit(AdStackLoadTopStmt *stmt) override;

  void visit(AdStackLoadTopAdjStmt *stmt) override;

  void visit(AdStackAccAdjointStmt *stmt) override;

  void visit(RangeAssumptionStmt *stmt) override;

  void visit(LoopUniqueStmt *stmt) override;

  void visit_call_bitcode(ExternalFuncCallStmt *stmt);

  void visit_call_shared_object(ExternalFuncCallStmt *stmt);

  void visit(ExternalFuncCallStmt *stmt) override;

  void visit(MeshPatchIndexStmt *stmt) override;

  void visit(ReferenceStmt *stmt) override;

  void visit(MatrixInitStmt *stmt) override;

  llvm::Value *create_xlogue(std::unique_ptr<Block> &block);

  llvm::Value *create_mesh_xlogue(std::unique_ptr<Block> &block);

  llvm::Value *extract_exponent_from_f32(llvm::Value *f);

  llvm::Value *extract_digits_from_f32(llvm::Value *f, bool full);

  llvm::Value *extract_digits_from_f32_with_shared_exponent(llvm::Value *f, llvm::Value *shared_exp);

  llvm::Value *get_exponent_offset(llvm::Value *exponent, QuantFloatType *qflt);

  void visit(FuncCallStmt *stmt) override;

  void visit(GetElementStmt *stmt) override;

  llvm::Value *bitcast_from_u64(llvm::Value *val, DataType type);
  llvm::Value *bitcast_to_u64(llvm::Value *val, DataType type);

  ~TaskCodeGenLLVM() override = default;

 private:
  void set_struct_to_buffer(llvm::Value *buffer,
                            llvm::Type *buffer_type,
                            const std::vector<Stmt *> &elements,
                            const Type *current_type,
                            int &current_element,
                            std::vector<llvm::Value *> &current_index);

  virtual std::tuple<llvm::Value *, llvm::Value *> get_spmd_info() = 0;
};

}  // namespace quadrants::lang

#endif  // #ifdef QD_WITH_LLVM
