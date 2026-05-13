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

namespace quadrants {
namespace lang {

using namespace llvm;

class TaskCodeGenAMDGPU : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;
  TaskCodeGenAMDGPU(int id,
                    const CompileConfig &config,
                    QuadrantsLLVMContext &tlctx,
                    const Kernel *kernel,
                    IRNode *ir = nullptr)
      : TaskCodeGenLLVM(id, config, tlctx, kernel, ir) {
  }

  llvm::Value *create_print(std::string tag, DataType dt, llvm::Value *value) override{QD_NOT_IMPLEMENTED}

  std::tuple<llvm::Value *, llvm::Type *> create_value_and_type(llvm::Value *value, DataType dt) {
    QD_NOT_IMPLEMENTED
  }

  void visit(PrintStmt *stmt) override {
    // We'll just ignore it
  }

  void emit_extra_unary(UnaryOpStmt *stmt) override {
    auto input = llvm_val[stmt->operand];
    auto input_quadrants_type = stmt->operand->ret_type;
    auto input_type = input->getType();
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
    }  // TODO simplify the impl of sgn
    else if (op == UnaryOpType::sgn) {
      if (input_quadrants_type->is_primitive(PrimitiveTypeID::i32)) {
        auto ashr = builder->CreateAShr(input, 31);
        auto sub = builder->CreateSub(0, input);
        auto lshr = builder->CreateLShr(sub, 31);
        llvm_val[stmt] = builder->CreateOr(ashr, lshr);
      } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::f32)) {
        auto func = builder->GetInsertBlock()->getParent();
        auto bb_oeq_then = BasicBlock::Create(*llvm_context, "oeq_then", func);
        auto bb_oeq_else = BasicBlock::Create(*llvm_context, "oeq_else");
        auto bb_merge = BasicBlock::Create(*llvm_context, "merge");
        auto bb_olt_then = BasicBlock::Create(*llvm_context, "olt_then", func);
        auto bb_olt_else = BasicBlock::Create(*llvm_context, "olt_else");

        auto alloc = builder->CreateAlloca(llvm::Type::getFloatTy(*llvm_context), (unsigned)5);
        auto newty = llvm::PointerType::get(llvm::Type::getFloatTy(*llvm_context), (unsigned)0);
        auto cast = builder->CreateAddrSpaceCast(alloc, newty);
        auto fcmp_oeq = builder->CreateFCmpOEQ(input, llvm::ConstantFP::get(llvm::Type::getFloatTy(*llvm_context), 0));
        builder->CreateCondBr(fcmp_oeq, bb_oeq_then, bb_oeq_else);
        builder->SetInsertPoint(bb_oeq_then);
        builder->CreateStore(llvm::ConstantFP::get(llvm::Type::getFloatTy(*llvm_context), 0), cast);
        builder->CreateBr(bb_merge);
        bb_oeq_then = builder->GetInsertBlock();

        func->insert(func->end(), bb_oeq_else);
        builder->SetInsertPoint(bb_oeq_else);
        auto fcmp_olt = builder->CreateFCmpOLT(input, llvm::ConstantFP::get(llvm::Type::getFloatTy(*llvm_context), 0));
        builder->CreateCondBr(fcmp_olt, bb_olt_then, bb_olt_else);
        bb_oeq_else = builder->GetInsertBlock();

        builder->SetInsertPoint(bb_olt_then);
        builder->CreateStore(llvm::ConstantFP::get(llvm::Type::getFloatTy(*llvm_context), -1), cast);
        builder->CreateBr(bb_merge);
        bb_olt_then = builder->GetInsertBlock();

        func->insert(func->end(), bb_olt_else);
        builder->SetInsertPoint(bb_olt_else);
        builder->CreateStore(llvm::ConstantFP::get(llvm::Type::getFloatTy(*llvm_context), 1), cast);
        builder->CreateBr(bb_merge);
        bb_olt_else = builder->GetInsertBlock();

        func->insert(func->end(), bb_merge);
        builder->SetInsertPoint(bb_merge);
        llvm_val[stmt] = builder->CreateLoad(llvm::Type::getFloatTy(*llvm_context), cast);
      } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::f64)) {
        auto func = builder->GetInsertBlock()->getParent();
        auto bb_oeq_then = BasicBlock::Create(*llvm_context, "oeq_then", func);
        auto bb_oeq_else = BasicBlock::Create(*llvm_context, "oeq_else");
        auto bb_merge = BasicBlock::Create(*llvm_context, "merge");
        auto bb_olt_then = BasicBlock::Create(*llvm_context, "olt_then", func);
        auto bb_olt_else = BasicBlock::Create(*llvm_context, "olt_else");

        auto alloc = builder->CreateAlloca(llvm::Type::getDoubleTy(*llvm_context), (unsigned)5);
        auto newty = llvm::PointerType::get(llvm::Type::getDoubleTy(*llvm_context), (unsigned)0);
        auto cast = builder->CreateAddrSpaceCast(alloc, newty);
        auto fcmp_oeq = builder->CreateFCmpOEQ(input, llvm::ConstantFP::get(llvm::Type::getDoubleTy(*llvm_context), 0));
        builder->CreateCondBr(fcmp_oeq, bb_oeq_then, bb_oeq_else);
        builder->SetInsertPoint(bb_oeq_then);
        builder->CreateStore(llvm::ConstantFP::get(llvm::Type::getDoubleTy(*llvm_context), 0), cast);
        builder->CreateBr(bb_merge);
        bb_oeq_then = builder->GetInsertBlock();

        func->insert(func->end(), bb_oeq_else);
        builder->SetInsertPoint(bb_oeq_else);
        auto fcmp_olt = builder->CreateFCmpOLT(input, llvm::ConstantFP::get(llvm::Type::getDoubleTy(*llvm_context), 0));
        builder->CreateCondBr(fcmp_olt, bb_olt_then, bb_olt_else);
        bb_oeq_else = builder->GetInsertBlock();

        builder->SetInsertPoint(bb_olt_then);
        builder->CreateStore(llvm::ConstantFP::get(llvm::Type::getDoubleTy(*llvm_context), -1), cast);
        builder->CreateBr(bb_merge);
        bb_olt_then = builder->GetInsertBlock();

        func->insert(func->end(), bb_olt_else);
        builder->SetInsertPoint(bb_olt_else);
        builder->CreateStore(llvm::ConstantFP::get(llvm::Type::getDoubleTy(*llvm_context), 1), cast);
        builder->CreateBr(bb_merge);
        bb_olt_else = builder->GetInsertBlock();

        func->insert(func->end(), bb_merge);
        builder->SetInsertPoint(bb_merge);
        llvm_val[stmt] = builder->CreateLoad(llvm::Type::getDoubleTy(*llvm_context), cast);
      }
    }
    UNARY_STD(cos)
    UNARY_STD(acos)
    UNARY_STD(sin)
    UNARY_STD(asin)
    UNARY_STD(tan)
    UNARY_STD(tanh)
    UNARY_STD(exp)
    UNARY_STD(log)
    UNARY_STD(sqrt)
    else if (op == UnaryOpType::popcnt) {
      // stmt->ret_type is already normalised to i32 by type_check.cpp; the explicit Trunc on the 64-bit arm keeps the
      // LLVM value width in sync with that contract.
      if (input_quadrants_type->is_primitive(PrimitiveTypeID::i32) ||
          input_quadrants_type->is_primitive(PrimitiveTypeID::u32)) {
        llvm_val[stmt] = builder->CreateIntrinsic(llvm::Intrinsic::ctpop, {input_type}, {input});
      } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::i64) ||
                 input_quadrants_type->is_primitive(PrimitiveTypeID::u64)) {
        auto pop64 = builder->CreateIntrinsic(llvm::Intrinsic::ctpop, {input_type}, {input});
        llvm_val[stmt] = builder->CreateTrunc(pop64, llvm::Type::getInt32Ty(*llvm_context));
      } else {
        QD_NOT_IMPLEMENTED
      }
    }
    else if (op == UnaryOpType::clz) {
      // clz operates on the unsigned bit pattern, so u32 / u64 lower to the same llvm.ctlz call as i32 / i64; LLVM IR
      // is signless for integers.
      auto is_zero_undef = llvm::ConstantInt::get(llvm::Type::getInt1Ty(*llvm_context), 0);
      if (input_quadrants_type->is_primitive(PrimitiveTypeID::i32) ||
          input_quadrants_type->is_primitive(PrimitiveTypeID::u32)) {
        llvm_val[stmt] = builder->CreateIntrinsic(llvm::Intrinsic::ctlz, {input_type}, {input, is_zero_undef});
      } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::i64) ||
                 input_quadrants_type->is_primitive(PrimitiveTypeID::u64)) {
        auto clz64 = builder->CreateIntrinsic(llvm::Intrinsic::ctlz, {input_type}, {input, is_zero_undef});
        llvm_val[stmt] = builder->CreateTrunc(clz64, llvm::Type::getInt32Ty(*llvm_context));
      } else {
        QD_NOT_IMPLEMENTED
      }
    }
    else if (op == UnaryOpType::ffs) {
      // ffs(x): 1-indexed position of the lowest set bit; 0 when x == 0 (CUDA __ffs convention). Lower to llvm.cttz + 1
      // and a select for the zero case; the AMDGPU LLVM backend further lowers llvm.cttz to native bitfield-extract
      // instructions. Same width-and-signedness gate as clz.
      auto is_zero_undef = llvm::ConstantInt::get(llvm::Type::getInt1Ty(*llvm_context), 0);
      if (input_quadrants_type->is_primitive(PrimitiveTypeID::i32) ||
          input_quadrants_type->is_primitive(PrimitiveTypeID::u32) ||
          input_quadrants_type->is_primitive(PrimitiveTypeID::i64) ||
          input_quadrants_type->is_primitive(PrimitiveTypeID::u64)) {
        auto cttz = builder->CreateIntrinsic(llvm::Intrinsic::cttz, {input_type}, {input, is_zero_undef});
        auto plus_one = builder->CreateAdd(cttz, llvm::ConstantInt::get(input_type, 1));
        auto is_zero = builder->CreateICmpEQ(input, llvm::ConstantInt::get(input_type, 0));
        auto sel = builder->CreateSelect(is_zero, llvm::ConstantInt::get(input_type, 0), plus_one);
        llvm_val[stmt] = builder->CreateZExtOrTrunc(sel, llvm::Type::getInt32Ty(*llvm_context));
      } else {
        QD_NOT_IMPLEMENTED
      }
    }
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
    PrimitiveTypeID prim_type = stmt->val->ret_type->cast<PrimitiveType>()->type;

    std::unordered_map<PrimitiveTypeID, std::unordered_map<AtomicOpType, std::string>> fast_reductions;

    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::add] = "reduce_add_i32";
    fast_reductions[PrimitiveTypeID::f32][AtomicOpType::add] = "reduce_add_f32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::min] = "reduce_min_i32";
    fast_reductions[PrimitiveTypeID::f32][AtomicOpType::min] = "reduce_min_f32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::max] = "reduce_max_i32";
    fast_reductions[PrimitiveTypeID::f32][AtomicOpType::max] = "reduce_max_f32";

    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_and] = "reduce_and_i32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_or] = "reduce_or_i32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_xor] = "reduce_xor_i32";

    AtomicOpType op = stmt->op_type;
    if (fast_reductions.find(prim_type) == fast_reductions.end()) {
      return nullptr;
    }
    QD_ASSERT(fast_reductions.at(prim_type).find(op) != fast_reductions.at(prim_type).end());
    return call(fast_reductions.at(prim_type).at(op), {llvm_val[stmt->dest], llvm_val[stmt->val]});
  }

  void visit(RangeForStmt *for_stmt) override {
    create_naive_range_for(for_stmt);
  }

  void create_offload_range_for(OffloadedStmt *stmt) override {
    auto tls_prologue = create_xlogue(stmt->tls_prologue);

    llvm::Function *body;
    {
      auto guard = get_function_creation_guard({llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
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
         {get_arg(0), begin, end, tls_prologue, body, epilogue, tlctx->get_constant(stmt->tls_size)});
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
    // on AMDGPU, pass the argument by value is not allowed
    return false;
  }

  void visit(GlobalLoadStmt *stmt) override {
    auto ptr = llvm_val[stmt->src];
    auto ptr_type = stmt->src->ret_type->as<PointerType>();
    if (ptr_type->is_bit_pointer()) {
      auto val_type = ptr_type->get_pointee_type();
      auto get_ch = stmt->src->as<GetChStmt>();
      auto physical_type = tlctx->get_data_type(get_ch->input_snode->physical_type);
      auto [byte_ptr, bit_offset] = load_bit_ptr(ptr);
      auto physical_value = builder->CreateLoad(physical_type, byte_ptr);
      if (auto qit = val_type->cast<QuantIntType>()) {
        llvm_val[stmt] = extract_quant_int(physical_value, bit_offset, qit);
      } else if (auto qfxt = val_type->cast<QuantFixedType>()) {
        qit = qfxt->get_digits_type()->as<QuantIntType>();
        auto digits = extract_quant_int(physical_value, bit_offset, qit);
        llvm_val[stmt] = reconstruct_quant_fixed(digits, qfxt);
      } else {
        QD_ASSERT(val_type->is<QuantFloatType>());
        QD_ASSERT(get_ch->input_snode->dt->is<BitStructType>());
        llvm_val[stmt] = extract_quant_float(physical_value, get_ch->input_snode->dt->as<BitStructType>(),
                                             get_ch->output_snode->id_in_bit_struct);
      }
    } else {
      // Byte pointer case.
      llvm_val[stmt] = builder->CreateLoad(tlctx->get_data_type(stmt->ret_type), ptr);
    }
  }

  void create_bls_buffer(OffloadedStmt *stmt) {
    QD_NOT_IMPLEMENTED
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
      // TODO
      // use amdgpu-jargons to replace nvidias'
      current_task->grid_dim = stmt->grid_dim;
      if (stmt->task_type == Type::range_for) {
        if (stmt->const_begin && stmt->const_end) {
          int num_threads = stmt->end_value - stmt->begin_value;
          int grid_dim = ((num_threads % stmt->block_dim) == 0) ? (num_threads / stmt->block_dim)
                                                                : (num_threads / stmt->block_dim) + 1;
          grid_dim = std::max(grid_dim, 1);
          current_task->grid_dim = std::min(stmt->grid_dim, grid_dim);
        }
      }
      if (stmt->task_type == Type::listgen) {
        // Note: 32 is a temporary number
        // TODO: find a func to obtain this attr
        int query_max_block_per_sm = 32;
        // AMDGPUDriver::get_instance().device_get_attribute(
        //     &query_max_block_per_sm,
        //     HIP_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, nullptr);
        int num_SMs;
        AMDGPUDriver::get_instance().device_get_attribute(&num_SMs, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
        current_task->grid_dim = num_SMs * query_max_block_per_sm;
      }
      current_task->block_dim = stmt->block_dim;
      current_task->stream_parallel_group_id = stmt->stream_parallel_group_id;
      QD_ASSERT(current_task->grid_dim != 0);
      QD_ASSERT(current_task->block_dim != 0);
      // Host-side adstack sizing, same scheme as codegen_cuda: tight `grid_dim * block_dim` for
      // non-range_for and const-bound range_for, dynamic resolution via gtmps DtoH memcpy for
      // dynamic-bound range_for. See llvm_compiled_data.h::AdStackSizingInfo for the resolution
      // rule the kernel launcher applies.
      if (current_task->ad_stack.per_thread_stride > 0) {
        current_task->ad_stack.static_num_threads =
            static_cast<std::size_t>(current_task->grid_dim) * static_cast<std::size_t>(current_task->block_dim);
        if (stmt->task_type == Type::range_for && !(stmt->const_begin && stmt->const_end)) {
          current_task->ad_stack.dynamic_gpu_range_for = true;
          current_task->ad_stack.begin_const_value = stmt->const_begin ? stmt->begin_value : 0;
          current_task->ad_stack.end_const_value = stmt->const_end ? stmt->end_value : 0;
          current_task->ad_stack.begin_offset_bytes =
              stmt->const_begin ? -1 : static_cast<std::int32_t>(stmt->begin_offset);
          current_task->ad_stack.end_offset_bytes = stmt->const_end ? -1 : static_cast<std::int32_t>(stmt->end_offset);
        }
      }
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
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

  void visit(InternalFuncStmt *stmt) override {
    if (stmt->func_name == "subgroupShuffle" || stmt->func_name == "subgroupBroadcast") {
      auto index = builder->CreateZExtOrTrunc(llvm_val[stmt->args[1]], llvm::Type::getInt32Ty(*llvm_context));
      llvm_val[stmt] = emit_amdgpu_shuffle(
          /* value=*/llvm_val[stmt->args[0]],
          /* dt=*/stmt->args[0]->ret_type, index);
    } else if (stmt->func_name == "subgroupShuffleDown") {
      auto offset = builder->CreateZExtOrTrunc(llvm_val[stmt->args[1]], llvm::Type::getInt32Ty(*llvm_context));
      llvm_val[stmt] = emit_amdgpu_shuffle_down(
          /* value=*/llvm_val[stmt->args[0]],
          /* dt=*/stmt->args[0]->ret_type, offset);
    } else if (stmt->func_name == "subgroupShuffleUp") {
      auto offset = builder->CreateZExtOrTrunc(llvm_val[stmt->args[1]], llvm::Type::getInt32Ty(*llvm_context));
      llvm_val[stmt] = emit_amdgpu_shuffle_up(
          /* value=*/llvm_val[stmt->args[0]],
          /* dt=*/stmt->args[0]->ret_type, offset);
    } else if (stmt->func_name == "subgroupBallotU32") {
      // We always lower to ``llvm.amdgcn.ballot.i64`` and truncate to i32, on both wave32 and wave64.  In principle
      // ``llvm.amdgcn.ballot.i32`` exists exactly for this case and is documented as well-defined on wave64 (PR
      // https://github.com/llvm/llvm-project/pull/71556 in LLVM 18: SETCC at wavefront width, then zext/trunc to the
      // requested return type, i.e. the low 32 bits = lanes 0..31's predicates on wave64).  In practice the LLVM
      // versions we've tested (20 and 22.1.0) still fail to select ``ballot.i32`` on gfx942 when the predicate is a
      // non-constant ``i1`` - isel hits "Cannot select: AMDGPUISD::SETCC ..." for the ``i1 -> i32 != 0`` predicate
      // shape that ``ballot_first_n`` produces in real kernels.  ``ballot.i64 + trunc to i32`` works around the bug
      // and produces identical assembly (same single ``v_cmp_*_e64`` + low-half store) since LLVM's CSE folds the
      // i64 ballot's high half away as soon as the trunc is observed.  See min repro in the PR thread; the workaround
      // costs nothing and is robust regardless of upstream LLVM fix status.
      auto ballot64 = call("amdgpu_ballot_u64", llvm_val[stmt->args[0]]);
      llvm_val[stmt] = builder->CreateTrunc(ballot64, llvm::Type::getInt32Ty(*llvm_context));
    } else if (stmt->func_name == "subgroupBallotU64") {
      // ``llvm.amdgcn.ballot.i64`` returns a 64-bit ballot for the full subgroup: on wave64 every lane contributes;
      // on wave32 only lanes 0..31 contribute and bits 32..63 of the result are zero.  Either way the i64 return is
      // uniform across wavefront modes, which is what ``ballot`` advertises to the user.  ``ballot.i64``
      // on either wave32 or wave64 selects cleanly in current LLVM (only the i32 form has the isel bug noted above).
      llvm_val[stmt] = call("amdgpu_ballot_u64", llvm_val[stmt->args[0]]);
    } else if (stmt->func_name == "subgroupInvocationId") {
      llvm_val[stmt] = call("amdgpu_lane_id");
    } else if (stmt->func_name == "subgroupBarrier") {
      // Wave-scope thread reconvergence barrier.  `llvm.amdgcn.wave.barrier` is the LLVM intrinsic AMDGPU exposes for
      // wave-level sync: on chips where waves are lockstep (GCN) it acts as a compiler reordering barrier; on RDNA it
      // lowers to a real wave-scope hardware barrier.  Caller contract is uniform CF + all lanes active.
      builder->CreateIntrinsic(Intrinsic::amdgcn_wave_barrier, ArrayRef<llvm::Value *>{});
      llvm_val[stmt] = tlctx->get_constant(0);
    } else if (stmt->func_name == "subgroupMemoryBarrier") {
      // Subgroup-scope memory fence.  AMDGPU has no first-class wave-scope memory fence intrinsic, so we emit an LLVM
      // `fence seq_cst` with workgroup syncscope.  The AMDGPU backend lowers this to the appropriate `s_waitcnt` /
      // cache-flush sequence.  Workgroup scope is over-strict for the subgroup-scope ask but correct (orders memory
      // across the whole workgroup, of which the subgroup is a subset) and matches what we do on CUDA
      // (`block_mem_fence`).
      builder->CreateFence(llvm::AtomicOrdering::SequentiallyConsistent,
                           llvm_context->getOrInsertSyncScopeID("workgroup"));
      llvm_val[stmt] = tlctx->get_constant(0);
    } else {
      TaskCodeGenLLVM::visit(stmt);
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
        auto sitofp_lhs_ = builder->CreateSIToFP(lhs, llvm::Type::getDoubleTy(*llvm_context));
        auto sitofp_rhs_ = builder->CreateSIToFP(rhs, llvm::Type::getDoubleTy(*llvm_context));
        auto ret_ = call("__ocml_pow_f64", {sitofp_lhs_, sitofp_rhs_});
        llvm_val[stmt] = builder->CreateFPToSI(ret_, llvm::Type::getInt32Ty(*llvm_context));
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
  llvm::Value *emit_amdgpu_shuffle(llvm::Value *value, DataType dt, llvm::Value *index) {
    if (dt->is_primitive(PrimitiveTypeID::i32) || dt->is_primitive(PrimitiveTypeID::u32))
      return call("amdgpu_shuffle_i32", index, value);
    if (dt->is_primitive(PrimitiveTypeID::f32))
      return call("amdgpu_shuffle_f32", index, value);
    if (dt->is_primitive(PrimitiveTypeID::f64))
      return call("amdgpu_shuffle_f64", index, value);
    if (dt->is_primitive(PrimitiveTypeID::i64) || dt->is_primitive(PrimitiveTypeID::u64))
      return call("amdgpu_shuffle_i64", index, value);
    QD_ERROR("subgroup shuffle: unsupported type {} on AMDGPU", data_type_name(dt));
    return nullptr;
  }

  // FIXME: Currently emulates shuffle_down via ds_bpermute (~50 cycle latency).
  // Should be upgraded to use DPP ROW_SHR instructions (~4-12 cycles) for
  // reduction-pattern offsets (1, 2, 4, 8, 16). This requires compile-time
  // constant DPP control values and architecture-specific handling for cross-row
  // shifts (offset >= 16).
  llvm::Value *emit_amdgpu_shuffle_down(llvm::Value *value, DataType dt, llvm::Value *offset) {
    if (dt->is_primitive(PrimitiveTypeID::i32) || dt->is_primitive(PrimitiveTypeID::u32))
      return call("amdgpu_shuffle_down_i32", offset, value);
    if (dt->is_primitive(PrimitiveTypeID::f32))
      return call("amdgpu_shuffle_down_f32", offset, value);
    if (dt->is_primitive(PrimitiveTypeID::f64))
      return call("amdgpu_shuffle_down_f64", offset, value);
    if (dt->is_primitive(PrimitiveTypeID::i64) || dt->is_primitive(PrimitiveTypeID::u64))
      return call("amdgpu_shuffle_down_i64", offset, value);
    QD_ERROR("subgroup shuffle_down: unsupported type {} on AMDGPU", data_type_name(dt));
    return nullptr;
  }

  // FIXME: Same DPP fast-path opportunity as `emit_amdgpu_shuffle_down` - currently emulates `shuffle_up` via
  // `ds_bpermute` (~50 cycle latency).
  llvm::Value *emit_amdgpu_shuffle_up(llvm::Value *value, DataType dt, llvm::Value *offset) {
    if (dt->is_primitive(PrimitiveTypeID::i32) || dt->is_primitive(PrimitiveTypeID::u32))
      return call("amdgpu_shuffle_up_i32", offset, value);
    if (dt->is_primitive(PrimitiveTypeID::f32))
      return call("amdgpu_shuffle_up_f32", offset, value);
    if (dt->is_primitive(PrimitiveTypeID::f64))
      return call("amdgpu_shuffle_up_f64", offset, value);
    if (dt->is_primitive(PrimitiveTypeID::i64) || dt->is_primitive(PrimitiveTypeID::u64))
      return call("amdgpu_shuffle_up_i64", offset, value);
    QD_ERROR("subgroup shuffle_up: unsupported type {} on AMDGPU", data_type_name(dt));
    return nullptr;
  }

  std::tuple<llvm::Value *, llvm::Value *> get_spmd_info() override {
    auto thread_idx = builder->CreateIntrinsic(Intrinsic::amdgcn_workitem_id_x, ArrayRef<llvm::Value *>{});
    auto workgroup_dim_ =
        call("__ockl_get_local_size", llvm::ConstantInt::get(llvm::Type::getInt32Ty(*llvm_context), 0));
    auto block_dim = builder->CreateTrunc(workgroup_dim_, llvm::Type::getInt32Ty(*llvm_context));
    return std::make_tuple(thread_idx, block_dim);
  }
};

LLVMCompiledTask KernelCodeGenAMDGPU::compile_task(int task_codegen_id,
                                                   const CompileConfig &config,
                                                   std::unique_ptr<llvm::Module> &&module,
                                                   IRNode *block) {
  TaskCodeGenAMDGPU gen(task_codegen_id, config, get_quadrants_llvm_context(), kernel, block);
  return gen.run_compilation();
}

}  // namespace lang
}  // namespace quadrants
