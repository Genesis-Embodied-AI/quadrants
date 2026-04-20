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

  struct AdStackSpirv {
    spirv::Value count_var;    // u32, Function scope - current number of entries
    spirv::Value primal_arr;   // Array<storage_type, max_size>, Function scope
    spirv::Value adjoint_arr;  // Array<storage_type, max_size>, Function scope
    // `elem_type` is the logical loop-carried value's SPIR-V type (e.g. bool for a u1 adstack). `storage_type`
    // is what the backing array is actually declared as: identical to `elem_type` except for u1, where the
    // array is declared as i32 because `IRBuilder::get_array_type` silently promotes OpTypeBool (which has no
    // defined storage layout under LogicalAddressing) to i32. Push/LoadTop/AccAdjoint must use `storage_type`
    // for the OpAccessChain / load-store pair, and cast between `elem_type` and `storage_type` around the
    // caller-visible value - otherwise SPIR-V codegen emits `OpAccessChain %_ptr_Function_bool %arr_of_int_N`,
    // which spirv-val rejects with "result type OpTypeBool does not match the type that results from
    // indexing into OpTypeInt" and AMD's native Vulkan driver runs anyway and segfaults the dispatch.
    spirv::SType elem_type;
    spirv::SType storage_type;
    uint32_t max_size{0};
  };
  std::unordered_map<const Stmt *, AdStackSpirv> ad_stacks_;
  spirv::Value ad_stack_access(spirv::Value arr, spirv::Value index, const spirv::SType &elem_type);
};
}  // namespace detail
}  // namespace spirv
}  // namespace quadrants::lang
