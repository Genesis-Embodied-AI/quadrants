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
    std::string ti_kernel_name;
    int task_id_in_kernel;
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
  spirv::Value load_buffer(const Stmt *ptr, DataType dt);
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
  // All existing type views of each underlying storage buffer, in creation order. When a second or later
  // view is minted in `get_buffer_value`, we decorate every entry here with `Aliased` so the driver is
  // forbidden from assuming the views don't alias -- otherwise a plain load through one view is not
  // ordered against an atomic through another view of the same memory, silently zeroing gradients on the
  // load-and-clear reverse-mode pattern. See `get_buffer_value` for the decoration site and the commit
  // message for the full failure matrix.
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

  // Where the primal/adjoint storage for an AdStack lives. `heap_float` backs f32 adstacks and `heap_int` backs
  // i32 and u1 adstacks (u1 stored as i32 to match the historical Function-scope path's bool->int remap in
  // `get_array_type`); other primitive types are hard-errored by `visit(AdStackAllocaStmt)`, so no Function-scope
  // fallback exists. Each kind maps to its own per-dispatch StorageBuffer (`BufferType::AdStackHeapFloat` /
  // `BufferType::AdStackHeapInt`).
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
  // Running offsets into the per-thread slice assigned to the next AdStackAllocaStmt visitor. Each ends equal to
  // the corresponding stride once every alloca has been visited; these feed the
  // `offset_in_elems_compile_time` of each alloca's `AdStackSizingAttribs::allocas` entry.
  uint32_t ad_stack_heap_next_offset_float_{0};
  uint32_t ad_stack_heap_next_offset_int_{0};
  // Buffers are cached for reuse across push/pop/load-top visitors and (re)computed lazily on first use inside a
  // task so the `OpLoad` falls inside the dispatch body rather than the function header.
  spirv::Value ad_stack_heap_buffer_float_;
  spirv::Value ad_stack_heap_buffer_int_;
  // `invoc_id * stride` thread-base values. Despite being cached like the buffers, these are NOT lazy: they are
  // emitted eagerly from `visit(AdStackAllocaStmt)` so the `OpIMul` lives in the alloca's enclosing block, which
  // strictly dominates every sibling inner loop that later references the cached SSA id. Emitting them lazily
  // from the first `AdStackPush/LoadTop` visitor would place the multiply in the first loop's body, and the
  // second sibling loop would reuse an SSA id defined in a non-dominating block (SPIR-V spec section 2.16).
  // Do NOT move these to a lazy path; the corresponding getters enforce eager emission.
  spirv::Value ad_stack_heap_thread_base_float_;
  spirv::Value ad_stack_heap_thread_base_int_;
  // Cached handle to the AdStackMetadata StorageBuffer and the per-task stride values loaded from
  // its header slots. Same dominance rule as the heap thread bases - eager emission at the first
  // alloca site of its heap kind, reused at every downstream push/load-top/load-top-adj.
  spirv::Value ad_stack_metadata_buffer_;
  spirv::Value ad_stack_metadata_stride_float_;
  spirv::Value ad_stack_metadata_stride_int_;
  // Return (lazily) the StorageBuffer of `Array<f32>` that backs f32 adstacks for this dispatch, and the
  // per-thread base index inside it.
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
  // Routes to the correct backing-typed pointer (`*f32` for `heap_float`, `*i32` for `heap_int`) based on
  // `info.heap_kind`. See comment on the implementation for the bool<->i32 conversion contract.
  spirv::Value ad_stack_slot_ptr(AdStackSpirv &info, spirv::Value idx, bool primal);
  spirv::SType ad_stack_backing_type(const AdStackSpirv &info) const;
};
}  // namespace detail
}  // namespace spirv
}  // namespace quadrants::lang
