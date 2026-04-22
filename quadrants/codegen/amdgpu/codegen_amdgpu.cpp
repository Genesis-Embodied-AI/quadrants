#include "quadrants/codegen/amdgpu/codegen_amdgpu.h"

#include <vector>
#include <set>
#include <functional>

#include "quadrants/common/core.h"
#include "quadrants/util/io.h"
#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/program/program.h"
#include "quadrants/util/lang_util.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/runtime/program_impls/llvm/llvm_program.h"
#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/codegen/codegen_utils.h"
#include "quadrants/inc/constants.h"

namespace quadrants {
namespace lang {

using namespace llvm;

class TaskCodeGenAMDGPU : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;
  size_t dynamic_shared_array_bytes{0};

  TaskCodeGenAMDGPU(int id,
                    const CompileConfig &config,
                    QuadrantsLLVMContext &tlctx,
                    const Kernel *kernel,
                    IRNode *ir = nullptr)
      : TaskCodeGenLLVM(id, config, tlctx, kernel, ir) {
  }

  llvm::Value *create_print(std::string tag,
                            DataType dt,
                            llvm::Value *value) override{QD_NOT_IMPLEMENTED}

  std::tuple<llvm::Value *, llvm::Type *> create_value_and_type(
      llvm::Value *value,
      DataType dt) {
    QD_NOT_IMPLEMENTED
  }

  void visit(PrintStmt *stmt) override {
    // We'll just ignore it
  }

  // Dynamic shared memory promotion
  void visit(AllocaStmt *stmt) override {
    auto tensor_type = stmt->ret_type.ptr_removed()->cast<TensorType>();
    if (tensor_type && stmt->is_shared) {
      size_t shared_array_bytes =
          tensor_type->get_num_elements() *
          data_type_size(tensor_type->get_element_type());
      if (shared_array_bytes > cuda_dynamic_shared_array_threshold_bytes) {
        if (dynamic_shared_array_bytes > 0) {
          QD_ERROR(
              "Only one single large shared array instance is allowed in "
              "current version.")
        }
        tensor_type->set_shape(std::vector<int>({0}));
        dynamic_shared_array_bytes += shared_array_bytes;
      }

      auto type = tlctx->get_data_type(tensor_type);
      auto base = new llvm::GlobalVariable(
          *module, type, false, llvm::GlobalValue::ExternalLinkage, nullptr,
          fmt::format("shared_array_t{}_s{}", task_codegen_id, stmt->id),
          nullptr, llvm::GlobalVariable::NotThreadLocal,
          3 /*addrspace=LDS*/);
      base->setAlignment(llvm::MaybeAlign(8));
      auto ptr_type = llvm::PointerType::get(type, 0);
      llvm_val[stmt] = builder->CreatePointerCast(base, ptr_type);
    } else {
      TaskCodeGenLLVM::visit(stmt);
    }
  }

  void emit_extra_unary(UnaryOpStmt *stmt) override {
    auto input = llvm_val[stmt->operand];
    auto input_quadrants_type = stmt->operand->ret_type;
    auto op = stmt->op_type;

#define UNARY_STD(x)                                                       \
  else if (op == UnaryOpType::x) {                                         \
    if (input_quadrants_type->is_primitive(PrimitiveTypeID::f16)) {        \
      llvm_val[stmt] = call("__ocml_" #x "_f16", input);                   \
    } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::f32)) { \
      llvm_val[stmt] = call("__ocml_" #x "_f32", input);                   \
    } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::f64)) { \
      llvm_val[stmt] = call("__ocml_" #x "_f64", input);                   \
    } else {                                                               \
      QD_NOT_IMPLEMENTED                                                   \
    }                                                                      \
  }
    if (op == UnaryOpType::abs) {
      if (input_quadrants_type->is_primitive(PrimitiveTypeID::f16)) {
        llvm_val[stmt] = call("__ocml_fasb_f16", input);
      } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__ocml_fabs_f32", input);
      } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__ocml_fabs_f64", input);
      } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::i32)) {
        auto ashr = builder->CreateAShr(input, 31);
        auto xor_i32 = builder->CreateXor(ashr, input);
        llvm_val[stmt] = builder->CreateSub(xor_i32, ashr, "", false, true);
      } else {
        QD_NOT_IMPLEMENTED
      }
    }
    // Branchless sgn using select (no alloca/scratch)
    else if (op == UnaryOpType::sgn) {
      if (input_quadrants_type->is_primitive(PrimitiveTypeID::i32)) {
        auto ashr = builder->CreateAShr(input, 31);
        auto sub = builder->CreateSub(0, input);
        auto lshr = builder->CreateLShr(sub, 31);
        llvm_val[stmt] = builder->CreateOr(ashr, lshr);
      } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::f32)) {
        auto *float_ty = llvm::Type::getFloatTy(*llvm_context);
        auto *zero = llvm::ConstantFP::get(float_ty, 0.0);
        auto *neg_one = llvm::ConstantFP::get(float_ty, -1.0);
        auto *pos_one = llvm::ConstantFP::get(float_ty, 1.0);
        auto *is_neg = builder->CreateFCmpOLT(input, zero);
        auto *is_zero = builder->CreateFCmpOEQ(input, zero);
        auto *neg_or_pos = builder->CreateSelect(is_neg, neg_one, pos_one);
        llvm_val[stmt] = builder->CreateSelect(is_zero, zero, neg_or_pos);
      } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::f64)) {
        auto *double_ty = llvm::Type::getDoubleTy(*llvm_context);
        auto *zero = llvm::ConstantFP::get(double_ty, 0.0);
        auto *neg_one = llvm::ConstantFP::get(double_ty, -1.0);
        auto *pos_one = llvm::ConstantFP::get(double_ty, 1.0);
        auto *is_neg = builder->CreateFCmpOLT(input, zero);
        auto *is_zero = builder->CreateFCmpOEQ(input, zero);
        auto *neg_or_pos = builder->CreateSelect(is_neg, neg_one, pos_one);
        llvm_val[stmt] = builder->CreateSelect(is_zero, zero, neg_or_pos);
      }
    }
    UNARY_STD(cos)
    UNARY_STD(sin)
    UNARY_STD(log)
    UNARY_STD(acos)
    UNARY_STD(asin)
    UNARY_STD(tan)
    UNARY_STD(tanh)
    UNARY_STD(exp)
    UNARY_STD(sqrt)
    else {
      QD_P(unary_op_type_name(op));
      QD_NOT_IMPLEMENTED
    }
#undef UNARY_STD
  }

  llvm::Value *optimized_reduction(AtomicOpStmt *stmt) override {
    if (!stmt->is_reduction) {
      return nullptr;
    }
    QD_ASSERT(stmt->val->ret_type->is<PrimitiveType>());
    PrimitiveTypeID prim_type =
        stmt->val->ret_type->cast<PrimitiveType>()->type;

    std::unordered_map<PrimitiveTypeID,
                       std::unordered_map<AtomicOpType, std::string>>
        fast_reductions;

    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::add] = "reduce_add_i32";
    fast_reductions[PrimitiveTypeID::f32][AtomicOpType::add] = "reduce_add_f32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::min] = "reduce_min_i32";
    fast_reductions[PrimitiveTypeID::f32][AtomicOpType::min] = "reduce_min_f32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::max] = "reduce_max_i32";
    fast_reductions[PrimitiveTypeID::f32][AtomicOpType::max] = "reduce_max_f32";

    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_and] =
        "reduce_and_i32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_or] =
        "reduce_or_i32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_xor] =
        "reduce_xor_i32";

    AtomicOpType op = stmt->op_type;
    if (fast_reductions.find(prim_type) == fast_reductions.end()) {
      return nullptr;
    }
    QD_ASSERT(fast_reductions.at(prim_type).find(op) !=
              fast_reductions.at(prim_type).end());
    // SNode pointer chain (GetRootStmt/SNodeLookupStmt/GetChStmt) propagates
    // addrspace(1) on AMDGPU. The runtime reduce_*_* helpers in
    // runtime.cpp:DEFINE_REDUCTION are declared with generic (addrspace 0)
    // pointer parameters. Cast the destination back to addrspace(0) so
    // check_func_call_signature accepts the call; InferAddressSpaces in O3
    // can re-promote downstream loads/stores after inlining.
    llvm::Value *dest = llvm_val[stmt->dest];
    if (dest && dest->getType()->isPointerTy() &&
        dest->getType()->getPointerAddressSpace() == 1) {
      auto *ptr_as0 = llvm::PointerType::getUnqual(*llvm_context);
      dest = builder->CreateAddrSpaceCast(dest, ptr_as0);
    }
    return call(fast_reductions.at(prim_type).at(op),
                {dest, llvm_val[stmt->val]});
  }

  void visit(RangeForStmt *for_stmt) override {
    create_naive_range_for(for_stmt);
  }

  void create_offload_range_for(OffloadedStmt *stmt) override {
    auto tls_prologue = create_xlogue(stmt->tls_prologue);

    llvm::Function *body;
    {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           get_tls_buffer_type(), tlctx->get_data_type<int>()});

      auto loop_var = create_entry_block_alloca(PrimitiveType::i32);
      loop_vars_llvm[stmt].push_back(loop_var);
      builder->CreateStore(get_arg(2), loop_var);
      stmt->body->accept(this);

      body = guard.body;
    }

    auto epilogue = create_xlogue(stmt->tls_epilogue);

    auto [begin, end] = get_range_for_bounds(stmt);
    call("gpu_parallel_range_for",
         {get_context(), begin, end, tls_prologue, body, epilogue,
          tlctx->get_constant(stmt->tls_size)});
  }

  void create_offload_mesh_for(OffloadedStmt *stmt) override {
    QD_NOT_IMPLEMENTED
  }

  void emit_amdgpu_gc(OffloadedStmt *stmt) {
    auto snode_id = tlctx->get_constant(stmt->snode->id);
    {
      init_offloaded_task_function(stmt, "gather_list");
      call("gc_parallel_0", get_context(), snode_id);
      finalize_offloaded_task_function();
      current_task->grid_dim = compile_config.saturating_grid_dim;
      current_task->block_dim = 64;
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
    {
      init_offloaded_task_function(stmt, "reinit_lists");
      call("gc_parallel_1", get_context(), snode_id);
      finalize_offloaded_task_function();
      current_task->grid_dim = 1;
      current_task->block_dim = 1;
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
    {
      init_offloaded_task_function(stmt, "zero_fill");
      call("gc_parallel_2", get_context(), snode_id);
      finalize_offloaded_task_function();
      current_task->grid_dim = compile_config.saturating_grid_dim;
      current_task->block_dim = 64;
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
  }

  bool kernel_argument_by_val() const override {
    return false;
  }

  bool kernel_argument_struct_in_kernarg() const override {
    return true;
  }

  // SNode root pointers are hipMalloc'd global memory. Cast result
  // to addrspace(1) so GEP chains produce global_load after inlining.
  void visit(GetRootStmt *stmt) override {
    TaskCodeGenLLVM::visit(stmt);
    auto *ptr_as1 = llvm::PointerType::get(*llvm_context, 1);
    llvm_val[stmt] = builder->CreateAddrSpaceCast(llvm_val[stmt], ptr_as1);
  }

  void visit(SNodeLookupStmt *stmt) override {
    // Cast addrspace(1) input to addrspace(0) for base visitor's
    // runtime function calls, then cast result back to addrspace(1).
    auto *input = llvm_val[stmt->input_snode];
    if (input && input->getType()->isPointerTy() &&
        input->getType()->getPointerAddressSpace() == 1) {
      auto *ptr_as0 = llvm::PointerType::getUnqual(*llvm_context);
      llvm_val[stmt->input_snode] =
          builder->CreateAddrSpaceCast(input, ptr_as0);
    }
    TaskCodeGenLLVM::visit(stmt);
    llvm_val[stmt->input_snode] = input;
    if (llvm_val[stmt] && llvm_val[stmt]->getType()->isPointerTy() &&
        llvm_val[stmt]->getType()->getPointerAddressSpace() == 0) {
      auto *ptr_as1 = llvm::PointerType::get(*llvm_context, 1);
      llvm_val[stmt] = builder->CreateAddrSpaceCast(llvm_val[stmt], ptr_as1);
    }
  }

  void visit(GetChStmt *stmt) override {
    if (stmt->input_snode->type == SNodeType::quant_array ||
        stmt->ret_type->as<PointerType>()->is_bit_pointer()) {
      TaskCodeGenLLVM::visit(stmt);
      return;
    }
    auto *input = llvm_val[stmt->input_ptr];
    if (input && input->getType()->isPointerTy() &&
        input->getType()->getPointerAddressSpace() == 1) {
      auto *ptr_as0 = llvm::PointerType::getUnqual(*llvm_context);
      llvm_val[stmt->input_ptr] =
          builder->CreateAddrSpaceCast(input, ptr_as0);
    }
    TaskCodeGenLLVM::visit(stmt);
    llvm_val[stmt->input_ptr] = input;
    if (llvm_val[stmt] && llvm_val[stmt]->getType()->isPointerTy() &&
        llvm_val[stmt]->getType()->getPointerAddressSpace() == 0) {
      auto *ptr_as1 = llvm::PointerType::get(*llvm_context, 1);
      llvm_val[stmt] = builder->CreateAddrSpaceCast(llvm_val[stmt], ptr_as1);
    }
  }

  llvm::Value *get_runtime() override {
    auto *runtime_context_ty = get_runtime_type("RuntimeContext");
    auto *runtime_ptr_addr = builder->CreateStructGEP(
        runtime_context_ty, TaskCodeGenLLVM::get_context(), 1);
    auto *runtime_ty =
        llvm::PointerType::get(get_runtime_type("LLVMRuntime"), 0);
    auto *runtime_ptr = builder->CreateLoad(runtime_ty, runtime_ptr_addr);
    auto *invariant_load_metadata =
        llvm::MDNode::get(builder->getContext(), {});
    runtime_ptr->setMetadata(llvm::LLVMContext::MD_invariant_load,
                             invariant_load_metadata);
    return runtime_ptr;
  }

  // Read-only cache loads via invariant.load metadata
  llvm::Value *create_intrinsic_load(llvm::Value *ptr,
                                     llvm::Type *ty) override {
    auto *ptr_ty_addrspace_1 = llvm::PointerType::get(ty, 1);
    auto *cast_ptr = builder->CreateAddrSpaceCast(ptr, ptr_ty_addrspace_1);
    auto *load = builder->CreateLoad(ty, cast_ptr);
    auto *invariant_load_metadata =
        llvm::MDNode::get(builder->getContext(), {});
    load->setMetadata(llvm::LLVMContext::MD_invariant_load,
                      invariant_load_metadata);
    return load;
  }

  // Predicate: returns true when the pointer source is local-storage-derived
  // (TLS scratch buffer or BLS / LDS shared array) and therefore must NOT be
  // addrspace(1)-cast. Walks through MatrixPtrStmt indirection to find the
  // underlying storage origin (matrix-typed TLS/BLS access).
  //
  // Background: Quadrants `GlobalLoadStmt`/`GlobalStoreStmt` cover any
  // pointer dereference — SNodes, external arrays (ndarray), TLS, BLS, and
  // global temporaries (statements.h:770-803). The first three of those plus
  // global temporaries genuinely live in hipMalloc'd global memory and
  // benefit from `addrspace(1)` (emits `global_load/store` instead of
  // `flat_load/store`, ~10-15 cycles saved per access on CDNA3 ISA §11.6;
  // -18.5% wall on monolith per CODE-VF-002). TLS and BLS do NOT — TLS is
  // an `addrspace(5)` scratch alloca passed as a flat pointer; BLS is
  // `addrspace(3)` LDS. Forcing those to `addrspace(1)` materializes
  // invalid global addresses and triggers
  // HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION at launch.
  static bool is_local_storage_source(Stmt *stmt) {
    while (stmt) {
      if (auto *m = stmt->cast<MatrixPtrStmt>()) {
        stmt = m->origin;
        continue;
      }
      break;
    }
    if (!stmt) {
      return false;
    }
    return stmt->is<ThreadLocalPtrStmt>() || stmt->is<BlockLocalPtrStmt>();
  }

  void visit(GlobalLoadStmt *stmt) override {
    auto ptr = llvm_val[stmt->src];
    auto ptr_type = stmt->src->ret_type->as<PointerType>();
    if (ptr_type->is_bit_pointer()) {
      if (auto get_ch = stmt->src->cast<GetChStmt>()) {
        bool should_cache_as_read_only =
            current_offload->mem_access_opt.has_flag(
                get_ch->output_snode, SNodeAccessFlag::read_only);
        create_global_load(stmt, should_cache_as_read_only);
      } else {
        create_global_load(stmt, false);
      }
    } else if (is_local_storage_source(stmt->src)) {
      TaskCodeGenLLVM::visit(stmt);
    } else {
      // Genuinely-global source (SNode / ExternalPtr / GlobalTemporary):
      // cast to addrspace(1) so LLVM emits global_load.
      auto *load_ty = tlctx->get_data_type(stmt->ret_type);
      bool read_only = false;
      if (auto get_ch = stmt->src->cast<GetChStmt>()) {
        read_only = current_offload->mem_access_opt.has_flag(
            get_ch->output_snode, SNodeAccessFlag::read_only);
      }
      auto *ptr_as1 = llvm::PointerType::get(load_ty, 1);
      auto *cast_ptr = builder->CreateAddrSpaceCast(ptr, ptr_as1);
      auto *load = builder->CreateLoad(load_ty, cast_ptr);
      if (read_only) {
        auto *md = llvm::MDNode::get(builder->getContext(), {});
        load->setMetadata(llvm::LLVMContext::MD_invariant_load, md);
      }
      llvm_val[stmt] = load;
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    QD_ASSERT(llvm_val[stmt->val]);
    QD_ASSERT(llvm_val[stmt->dest]);
    auto ptr_type = stmt->dest->ret_type->as<PointerType>();
    if (ptr_type->is_bit_pointer()) {
      TaskCodeGenLLVM::visit(stmt);
    } else if (is_local_storage_source(stmt->dest)) {
      TaskCodeGenLLVM::visit(stmt);
    } else {
      // Genuinely-global dest: cast to addrspace(1) for global_store.
      auto *val_ty = llvm_val[stmt->val]->getType();
      auto *ptr_as1 = llvm::PointerType::get(val_ty, 1);
      auto *cast_ptr =
          builder->CreateAddrSpaceCast(llvm_val[stmt->dest], ptr_as1);
      builder->CreateStore(llvm_val[stmt->val], cast_ptr);
    }
  }

  // BLS / shared memory buffer allocation
  void create_bls_buffer(OffloadedStmt *stmt) {
    auto type = llvm::ArrayType::get(
        llvm::Type::getInt8Ty(*llvm_context), stmt->bls_size);
    bls_buffer = new llvm::GlobalVariable(
        *module, type, false, llvm::GlobalValue::ExternalLinkage, nullptr,
        "bls_buffer", nullptr, llvm::GlobalVariable::NotThreadLocal,
        3 /*addrspace=LDS*/);
    bls_buffer->setAlignment(llvm::MaybeAlign(8));
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->bls_size > 0)
      create_bls_buffer(stmt);
#if defined(QD_WITH_AMDGPU)
    QD_ASSERT(current_offload == nullptr);
    current_offload = stmt;
    using Type = OffloadedStmt::TaskType;
    if (stmt->task_type == Type::gc) {
      emit_amdgpu_gc(stmt);
    } else {
      init_offloaded_task_function(stmt);
      if (stmt->task_type == Type::serial) {
        stmt->body->accept(this);
      } else if (stmt->task_type == Type::range_for) {
        create_offload_range_for(stmt);
      } else if (stmt->task_type == Type::struct_for) {
        create_offload_struct_for(stmt);
      } else if (stmt->task_type == Type::mesh_for) {
        create_offload_mesh_for(stmt);
      } else if (stmt->task_type == Type::listgen) {
        emit_list_gen(stmt);
      } else {
        QD_NOT_IMPLEMENTED
      }
      finalize_offloaded_task_function();
      // Wavefront size is 64.  Workgroups smaller than the wavefront
      // size are not handled reliably by the HSA runtime when the kernel
      // uses scratch memory (which we do via addrspacecast'd alloca's).
      // In that case the kernel launch fails with
      // HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION / hipErrorIllegalAddress.
      constexpr int kAmdgpuWavefrontSize = 64;
      int effective_block_dim = stmt->block_dim;
      if ((stmt->task_type == Type::range_for ||
           stmt->task_type == Type::struct_for ||
           stmt->task_type == Type::mesh_for) &&
          effective_block_dim > 0 &&
          effective_block_dim < kAmdgpuWavefrontSize) {
        effective_block_dim = kAmdgpuWavefrontSize;
      }

      current_task->grid_dim = stmt->grid_dim;
      if (stmt->task_type == Type::range_for) {
        if (stmt->const_begin && stmt->const_end) {
          int num_threads = stmt->end_value - stmt->begin_value;
          int grid_dim = ((num_threads % effective_block_dim) == 0)
                             ? (num_threads / effective_block_dim)
                             : (num_threads / effective_block_dim) + 1;
          grid_dim = std::max(grid_dim, 1);
          current_task->grid_dim = std::min(stmt->grid_dim, grid_dim);
        }
      }
      if (stmt->task_type == Type::listgen) {
        int num_SMs;
        AMDGPUDriver::get_instance().device_get_attribute(
            &num_SMs, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
        int max_threads_per_sm = 0;
        AMDGPUDriver::get_instance().device_get_attribute(
            &max_threads_per_sm,
            HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, 0);
        int query_max_block_per_sm =
            (max_threads_per_sm > 0 && effective_block_dim > 0)
                ? (max_threads_per_sm / effective_block_dim)
                : 32;
        current_task->grid_dim = num_SMs * query_max_block_per_sm;
      }
      current_task->block_dim = effective_block_dim;
      current_task->dynamic_shared_array_bytes = dynamic_shared_array_bytes;
      QD_ASSERT(current_task->grid_dim != 0);
      QD_ASSERT(current_task->block_dim != 0);
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
      dynamic_shared_array_bytes = 0;
    }
    current_offload = nullptr;
#else
    QD_NOT_IMPLEMENTED
#endif
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    if (stmt->type == ExternalFuncCallStmt::BITCODE) {
      TaskCodeGenLLVM::visit_call_bitcode(stmt);
    } else {
      QD_NOT_IMPLEMENTED
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    auto op = stmt->op_type;
    auto ret_quadrants_type = stmt->ret_type;
    if (op != BinaryOpType::atan2 && op != BinaryOpType::pow) {
      return TaskCodeGenLLVM::visit(stmt);
    }
    auto lhs = llvm_val[stmt->lhs];
    auto rhs = llvm_val[stmt->rhs];

    if (op == BinaryOpType::pow) {
      if (ret_quadrants_type->is_primitive(PrimitiveTypeID::f16)) {
        llvm_val[stmt] = call("__ocml_pow_f16", {lhs, rhs});
      } else if (ret_quadrants_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__ocml_pow_f32", {lhs, rhs});
      } else if (ret_quadrants_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__ocml_pow_f64", {lhs, rhs});
      } else if (ret_quadrants_type->is_primitive(PrimitiveTypeID::i32)) {
        auto sitofp_lhs_ =
            builder->CreateSIToFP(lhs, llvm::Type::getDoubleTy(*llvm_context));
        auto sitofp_rhs_ =
            builder->CreateSIToFP(rhs, llvm::Type::getDoubleTy(*llvm_context));
        auto ret_ = call("__ocml_pow_f64", {sitofp_lhs_, sitofp_rhs_});
        llvm_val[stmt] =
            builder->CreateFPToSI(ret_, llvm::Type::getInt32Ty(*llvm_context));
      } else {
        QD_NOT_IMPLEMENTED
      }
    } else if (op == BinaryOpType::atan2) {
      if (ret_quadrants_type->is_primitive(PrimitiveTypeID::f16)) {
        llvm_val[stmt] = call("__ocml_atan2_f16", {lhs, rhs});
      } else if (ret_quadrants_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__ocml_atan2_f32", {lhs, rhs});
      } else if (ret_quadrants_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__ocml_atan2_f64", {lhs, rhs});
      } else {
        QD_NOT_IMPLEMENTED
      }
    }
  }

 private:
  std::tuple<llvm::Value *, llvm::Value *> get_spmd_info() override {
    auto thread_idx = builder->CreateIntrinsic(Intrinsic::amdgcn_workitem_id_x,
                                               ArrayRef<llvm::Value *>{});
    auto workgroup_dim_ =
        call("__ockl_get_local_size",
             llvm::ConstantInt::get(llvm::Type::getInt32Ty(*llvm_context), 0));
    auto block_dim = builder->CreateTrunc(
        workgroup_dim_, llvm::Type::getInt32Ty(*llvm_context));
    return std::make_tuple(thread_idx, block_dim);
  }
};

LLVMCompiledTask KernelCodeGenAMDGPU::compile_task(
    int task_codegen_id,
    const CompileConfig &config,
    std::unique_ptr<llvm::Module> &&module,
    IRNode *block) {
  TaskCodeGenAMDGPU gen(task_codegen_id, config, get_quadrants_llvm_context(),
                        kernel, block);
  return gen.run_compilation();
}

}  // namespace lang
}  // namespace quadrants
