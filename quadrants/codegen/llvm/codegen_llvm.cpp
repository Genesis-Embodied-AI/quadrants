#include "quadrants/codegen/llvm/codegen_llvm.h"

#include <algorithm>

#ifdef QD_WITH_LLVM
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/snode.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/extension.h"
#include "quadrants/runtime/program_impls/llvm/llvm_program.h"
#include "quadrants/codegen/llvm/struct_llvm.h"
#include "quadrants/util/file_sequence_writer.h"
#include "quadrants/codegen/codegen_utils.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/AsmParser/Parser.h"
#include "quadrants/codegen/ir_dump.h"
#include "quadrants/util/environ_config.h"
#include "quadrants/runtime/llvm/llvm_context_pass.h"
#include "quadrants/runtime/llvm/kernel_atomic_syncscope.h"

namespace quadrants::lang {

// TODO: sort function definitions to match declaration order in header

// TODO(k-ye): Hide FunctionCreationGuard inside cpp file
FunctionCreationGuard::FunctionCreationGuard(TaskCodeGenLLVM *mb,
                                             std::vector<llvm::Type *> arguments,
                                             const std::string &func_name)
    : mb(mb) {
  // Create the loop body function
  auto body_function_type = llvm::FunctionType::get(llvm::Type::getVoidTy(*mb->llvm_context), arguments, false);

  body = llvm::Function::Create(body_function_type, llvm::Function::InternalLinkage, func_name, mb->module.get());
  old_func = mb->func;
  // emit into loop body function
  mb->func = body;

  allocas = llvm::BasicBlock::Create(*mb->llvm_context, "allocs", body);
  old_entry = mb->entry_block;
  mb->entry_block = allocas;

  final = llvm::BasicBlock::Create(*mb->llvm_context, "final", body);
  old_final = mb->final_block;
  mb->final_block = final;

  entry = llvm::BasicBlock::Create(*mb->llvm_context, "entry", mb->func);

  ip = mb->builder->saveIP();
  mb->builder->SetInsertPoint(entry);

  auto body_bb = llvm::BasicBlock::Create(*mb->llvm_context, "function_body", mb->func);
  mb->builder->CreateBr(body_bb);
  mb->builder->SetInsertPoint(body_bb);
}

FunctionCreationGuard::~FunctionCreationGuard() {
  if (!mb->returned) {
    mb->builder->CreateBr(final);
  }
  mb->builder->SetInsertPoint(final);
  mb->builder->CreateRetVoid();
  mb->returned = false;

  mb->builder->SetInsertPoint(allocas);
  mb->builder->CreateBr(entry);

  mb->entry_block = old_entry;
  mb->final_block = old_final;
  mb->func = old_func;
  mb->builder->restoreIP(ip);

  QD_ASSERT(!llvm::verifyFunction(*body, &llvm::errs()));
}

namespace {

class CodeGenStmtGuard {
 public:
  using Getter = std::function<llvm::BasicBlock *(void)>;
  using Setter = std::function<void(llvm::BasicBlock *)>;

  explicit CodeGenStmtGuard(Getter getter, Setter setter) : saved_stmt_(getter()), setter_(std::move(setter)) {
  }

  ~CodeGenStmtGuard() {
    setter_(saved_stmt_);
  }

  CodeGenStmtGuard(CodeGenStmtGuard &&) = default;
  CodeGenStmtGuard &operator=(CodeGenStmtGuard &&) = default;

 private:
  llvm::BasicBlock *saved_stmt_;
  Setter setter_;
};

CodeGenStmtGuard make_loop_reentry_guard(TaskCodeGenLLVM *cg) {
  return CodeGenStmtGuard([cg]() { return cg->current_loop_reentry; },
                          [cg](llvm::BasicBlock *saved_stmt) { cg->current_loop_reentry = saved_stmt; });
}

CodeGenStmtGuard make_while_after_loop_guard(TaskCodeGenLLVM *cg) {
  return CodeGenStmtGuard([cg]() { return cg->current_while_after_loop; },
                          [cg](llvm::BasicBlock *saved_stmt) { cg->current_while_after_loop = saved_stmt; });
}

}  // namespace

// TaskCodeGenLLVM
void TaskCodeGenLLVM::visit(Block *stmt_list) {
  // Float-heap lazy row claim at the IR-level Lowest Common Ancestor (LCA) of every f32 push / load-top site. Mirrors
  // the SPIR-V codegen's `visit(Block *)` pivot. Active only when the shared static analysis captured a gating
  // `bound_expr` for this task and resolved a non-trivial LCA: tasks without a captured gate keep the legacy
  // combined-heap eager addressing and never enter this branch. The runtime-side counter
  // (`runtime->adstack_row_counters[task_codegen_id]`) and capacity (`adstack_bound_row_capacities`) arrays the
  // atomicrmw and clamp read against are allocated and reset by every launcher (CPU / CUDA / AMDGPU) before the first
  // task in a kernel via `publish_adstack_lazy_claim_buffers`, so the claim is safe to fire.
  if (ad_stack_static_bound_expr_.has_value() && ad_stack_lca_block_float_ir_ != nullptr &&
      stmt_list == ad_stack_lca_block_float_ir_) {
    emit_ad_stack_row_claim_llvm();
    if (compile_config.debug) {
      // Debug build: route the heap-header `stack_init` (writes the u64 count word at offset 0) through the
      // freshly-claimed row so the first `stack_push` reads count = 0. The alloca-site path skipped this call
      // intentionally - at that IR position `row_id_var` was still its UINT32_MAX entry-block init, so
      // `get_ad_stack_base_llvm(stack)` would have addressed off the heap. Now that the LCA-block atomic-rmw has stored
      // the per-thread row id we can safely materialise the per-stack base and zero its header.
      for (AdStackAllocaStmt *lazy_stmt : ad_stack_lazy_float_allocas_) {
        call("stack_init", get_ad_stack_base_llvm(lazy_stmt));
      }
    }
  }
  for (auto &stmt : stmt_list->statements) {
    stmt->accept(this);
    if (returned) {
      break;
    }
  }
}

void TaskCodeGenLLVM::visit(AllocaStmt *stmt) {
  auto alloca_type = stmt->ret_type.ptr_removed();
  if (alloca_type->is<TensorType>()) {
    auto tensor_type = alloca_type->cast<TensorType>();
    auto type = tlctx->get_data_type(tensor_type);
    if (stmt->is_shared) {
      auto base = new llvm::GlobalVariable(*module, type, false, llvm::GlobalValue::ExternalLinkage, nullptr,
                                           fmt::format("shared_array_t{}_s{}", task_codegen_id, stmt->id), nullptr,
                                           llvm::GlobalVariable::NotThreadLocal, 3 /*addrspace=shared*/);
      base->setAlignment(llvm::MaybeAlign(8));
      auto ptr_type = llvm::PointerType::get(type, 0);
      llvm_val[stmt] = builder->CreatePointerCast(base, ptr_type);
    } else {
      llvm_val[stmt] = create_entry_block_alloca(type);
    }
  } else {
    llvm_val[stmt] = create_entry_block_alloca(alloca_type);
    // initialize as zero if element is not a pointer
    if (!alloca_type->is<PointerType>())
      builder->CreateStore(tlctx->get_constant(alloca_type, 0), llvm_val[stmt]);
  }
}

void TaskCodeGenLLVM::visit(RandStmt *stmt) {
  if (stmt->ret_type->is_primitive(PrimitiveTypeID::f16)) {
    // Promoting to f32 since there's no rand_f16 support in runtime.cpp.
    auto val_f32 = call("rand_f32", get_context());
    llvm_val[stmt] = builder->CreateFPTrunc(val_f32, llvm::Type::getHalfTy(*llvm_context));
  } else {
    llvm_val[stmt] = call(fmt::format("rand_{}", data_type_name(stmt->ret_type)), get_context());
  }
}

void TaskCodeGenLLVM::emit_extra_unary(UnaryOpStmt *stmt) {
  auto input = llvm_val[stmt->operand];
  auto input_quadrants_type = stmt->operand->ret_type;
  if (input_quadrants_type->is_primitive(PrimitiveTypeID::f16)) {
    // Promote to f32 since we don't have f16 support for extra unary ops in in
    // runtime.cpp.
    input = builder->CreateFPExt(input, llvm::Type::getFloatTy(*llvm_context));
    input_quadrants_type = PrimitiveType::f32;
  }

  auto op = stmt->op_type;
  auto input_type = input->getType();

#define UNARY_STD(x)                                                       \
  else if (op == UnaryOpType::x) {                                         \
    if (input_quadrants_type->is_primitive(PrimitiveTypeID::f32)) {        \
      llvm_val[stmt] = call(#x "_f32", input);                             \
    } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::f64)) { \
      llvm_val[stmt] = call(#x "_f64", input);                             \
    } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::i32)) { \
      llvm_val[stmt] = call(#x "_i32", input);                             \
    } else if (input_quadrants_type->is_primitive(PrimitiveTypeID::i64)) { \
      llvm_val[stmt] = call(#x "_i64", input);                             \
    } else {                                                               \
      QD_NOT_IMPLEMENTED                                                   \
    }                                                                      \
  }
  if (false) {
  }
  UNARY_STD(abs)
  UNARY_STD(exp)
  UNARY_STD(log)
  UNARY_STD(tan)
  UNARY_STD(tanh)
  UNARY_STD(sgn)
  UNARY_STD(acos)
  UNARY_STD(asin)
  UNARY_STD(cos)
  UNARY_STD(sin)
  else if (op == UnaryOpType::sqrt) {
    llvm_val[stmt] = builder->CreateIntrinsic(llvm::Intrinsic::sqrt, {input_type}, {input});
  }
  else if (op == UnaryOpType::popcnt) {
    // stmt->ret_type is already normalised to i32 by type_check.cpp; the explicit truncation here keeps the LLVM
    // value width in sync with that contract on 64-bit operands.
    auto pop = builder->CreateIntrinsic(llvm::Intrinsic::ctpop, {input_type}, {input});
    llvm_val[stmt] = builder->CreateZExtOrTrunc(pop, llvm::Type::getInt32Ty(*llvm_context));
  }
  else if (op == UnaryOpType::clz) {
    auto clz = builder->CreateIntrinsic(llvm::Intrinsic::ctlz, {input_type},
                                        {input, llvm::ConstantInt::get(llvm::Type::getInt1Ty(*llvm_context), 0)});
    llvm_val[stmt] = builder->CreateZExtOrTrunc(clz, llvm::Type::getInt32Ty(*llvm_context));
  }
  else if (op == UnaryOpType::ffs) {
    // ffs(x): 1-indexed position of the lowest set bit; 0 when x == 0 (CUDA __ffs convention). llvm.cttz with
    // is_zero_undef = false returns bitwidth on a zero input, so we explicitly select 0 for that case rather than
    // letting the +1 produce bitwidth + 1.
    auto is_zero_undef = llvm::ConstantInt::get(llvm::Type::getInt1Ty(*llvm_context), 0);
    auto cttz = builder->CreateIntrinsic(llvm::Intrinsic::cttz, {input_type}, {input, is_zero_undef});
    auto plus_one = builder->CreateAdd(cttz, llvm::ConstantInt::get(input_type, 1));
    auto is_zero = builder->CreateICmpEQ(input, llvm::ConstantInt::get(input_type, 0));
    auto sel = builder->CreateSelect(is_zero, llvm::ConstantInt::get(input_type, 0), plus_one);
    llvm_val[stmt] = builder->CreateZExtOrTrunc(sel, llvm::Type::getInt32Ty(*llvm_context));
  }
  else {
    QD_P(unary_op_type_name(op));
    QD_NOT_IMPLEMENTED
  }
#undef UNARY_STD
  if (stmt->ret_type->is_primitive(PrimitiveTypeID::f16)) {
    // Convert back to f16
    llvm_val[stmt] = builder->CreateFPTrunc(llvm_val[stmt], llvm::Type::getHalfTy(*llvm_context));
  }
}

std::unique_ptr<RuntimeObject> TaskCodeGenLLVM::emit_struct_meta_object(SNode *snode) {
  std::unique_ptr<RuntimeObject> meta;
  if (snode->type == SNodeType::dense) {
    meta = std::make_unique<RuntimeObject>("DenseMeta", this, builder.get());
    emit_struct_meta_base("Dense", meta->ptr, snode);
    meta->call("set_morton_dim", tlctx->get_constant((int)snode->_morton));
  } else if (snode->type == SNodeType::pointer) {
    meta = std::make_unique<RuntimeObject>("PointerMeta", this, builder.get());
    emit_struct_meta_base("Pointer", meta->ptr, snode);
  } else if (snode->type == SNodeType::root) {
    meta = std::make_unique<RuntimeObject>("RootMeta", this, builder.get());
    emit_struct_meta_base("Root", meta->ptr, snode);
  } else if (snode->type == SNodeType::dynamic) {
    meta = std::make_unique<RuntimeObject>("DynamicMeta", this, builder.get());
    emit_struct_meta_base("Dynamic", meta->ptr, snode);
    meta->call("set_chunk_size", tlctx->get_constant(snode->chunk_size));
  } else if (snode->type == SNodeType::bitmasked) {
    meta = std::make_unique<RuntimeObject>("BitmaskedMeta", this, builder.get());
    emit_struct_meta_base("Bitmasked", meta->ptr, snode);
  } else if (snode->type == SNodeType::quant_array) {
    meta = std::make_unique<RuntimeObject>("DenseMeta", this, builder.get());
    emit_struct_meta_base("Dense", meta->ptr, snode);
  } else {
    QD_P(snode_type_name(snode->type));
    QD_NOT_IMPLEMENTED;
  }
  return meta;
}

void TaskCodeGenLLVM::emit_struct_meta_base(const std::string &name, llvm::Value *node_meta, SNode *snode) {
  RuntimeObject common("StructMeta", this, builder.get(), node_meta);
  std::size_t element_size;
  if (snode->type == SNodeType::dense) {
    auto body_type = StructCompilerLLVM::get_llvm_body_type(module.get(), snode);
    auto element_ty = body_type->getArrayElementType();
    element_size = tlctx->get_type_size(element_ty);
  } else if (snode->type == SNodeType::pointer) {
    auto element_ty = StructCompilerLLVM::get_llvm_node_type(module.get(), snode->ch[0].get());
    element_size = tlctx->get_type_size(element_ty);
  } else {
    auto element_ty = StructCompilerLLVM::get_llvm_element_type(module.get(), snode);
    element_size = tlctx->get_type_size(element_ty);
  }
  common.set("snode_id", tlctx->get_constant(snode->id));
  common.set("element_size", tlctx->get_constant((uint64)element_size));
  common.set("max_num_elements", tlctx->get_constant(snode->max_num_elements()));
  common.set("context", get_context());

  /*
  uint8 *(*lookup_element)(uint8 *, int i);
  uint8 *(*from_parent_element)(uint8 *);
  bool (*is_active)(uint8 *, int i);
  int (*get_num_elements)(uint8 *);
  void (*refine_coordinates)(PhysicalCoordinates *inp_coord,
                             PhysicalCoordinates *refined_coord,
                             int index);
                             */

  std::vector<std::string> functions = {"lookup_element", "is_active", "get_num_elements"};

  for (auto const &f : functions)
    common.set(f, get_runtime_function(fmt::format("{}_{}", name, f)));

  // "from_parent_element", "refine_coordinates" are different for different
  // snodes, even if they have the same type.
  if (snode->parent)
    common.set("from_parent_element",
               get_struct_function(snode->get_ch_from_parent_func_name(), snode->get_snode_tree_id()));

  if (snode->type != SNodeType::place)
    common.set("refine_coordinates",
               get_struct_function(snode->refine_coordinates_func_name(), snode->get_snode_tree_id()));
}

TaskCodeGenLLVM::TaskCodeGenLLVM(int id,
                                 const CompileConfig &compile_config,
                                 QuadrantsLLVMContext &tlctx,
                                 const Kernel *kernel,
                                 IRNode *ir,
                                 std::unique_ptr<llvm::Module> &&module)
    // TODO: simplify LLVMModuleBuilder ctor input
    : LLVMModuleBuilder(module == nullptr ? tlctx.new_module("kernel") : std::move(module), &tlctx),
      compile_config(compile_config),
      kernel(kernel),
      ir(ir),
      prog(kernel->program),
      task_codegen_id(id) {
  if (ir == nullptr)
    this->ir = kernel->ir.get();
  initialize_context();

  context_ty = get_runtime_type("RuntimeContext");
  physical_coordinate_ty = get_runtime_type(kLLVMPhysicalCoordinatesName);

  kernel_name = kernel->name + "_kernel";
  current_callable = kernel;
}

void TaskCodeGenLLVM::visit(DecorationStmt *stmt) {
}

void TaskCodeGenLLVM::create_elementwise_cast(UnaryOpStmt *stmt,
                                              llvm::Type *to_ty,
                                              std::function<llvm::Value *(llvm::Value *, llvm::Type *)> func,
                                              bool on_self) {
  auto from_ty = stmt->operand->ret_type->cast<TensorType>();
  QD_ASSERT_INFO(from_ty, "Cannot perform elementwise ops on non-tensor type {}", from_ty->to_string());
  llvm::Value *vec =
      llvm::UndefValue::get(llvm::VectorType::get(to_ty, from_ty->get_num_elements(), /*scalable=*/false));
  for (int i = 0; i < from_ty->get_num_elements(); ++i) {
    auto elem = builder->CreateExtractElement(on_self ? llvm_val[stmt] : llvm_val[stmt->operand], i);
    auto cast_value = func(elem, to_ty);
    vec = builder->CreateInsertElement(vec, cast_value, i);
  }
  llvm_val[stmt] = vec;
}

void TaskCodeGenLLVM::visit(UnaryOpStmt *stmt) {
  auto input = llvm_val[stmt->operand];
  auto input_type = input->getType();
  auto op = stmt->op_type;

#define UNARY_INTRINSIC(x)                                                                \
  else if (op == UnaryOpType::x) {                                                        \
    llvm_val[stmt] = builder->CreateIntrinsic(llvm::Intrinsic::x, {input_type}, {input}); \
  }
  if (stmt->op_type == UnaryOpType::cast_value) {
    llvm::CastInst::CastOps cast_op;
    auto from = stmt->operand->ret_type;
    auto to = stmt->ret_type;
    QD_ASSERT_INFO(from->is<TensorType>() == to->is<TensorType>(),
                   "Cannot cast between tensor type and non-tensor type: {} v.s. {}", from->to_string(),
                   to->to_string());

    if (from == to) {
      llvm_val[stmt] = llvm_val[stmt->operand];
    } else if (to->is_primitive(PrimitiveTypeID::u1)) {
      llvm_val[stmt] = builder->CreateIsNotNull(input);
    } else if (is_real(from.get_element_type()) != is_real(to.get_element_type())) {
      if (is_real(from.get_element_type()) && (is_integral(to.get_element_type()))) {
        cast_op = (is_signed(to.get_element_type())) ? llvm::Instruction::CastOps::FPToSI
                                                     : llvm::Instruction::CastOps::FPToUI;
      } else if (is_integral(from.get_element_type()) && is_real(to.get_element_type())) {
        cast_op = (is_signed(from.get_element_type())) ? llvm::Instruction::CastOps::SIToFP
                                                       : llvm::Instruction::CastOps::UIToFP;
      } else {
        QD_P(data_type_name(from));
        QD_P(data_type_name(to));
        QD_NOT_IMPLEMENTED;
      }
      bool use_f16 =
          to->is_primitive(PrimitiveTypeID::f16) ||
          (to->is<TensorType>() && to->cast<TensorType>()->get_element_type()->is_primitive(PrimitiveTypeID::f16));
      auto cast_type = use_f16 ? (to->is<TensorType>() ? TypeFactory::create_tensor_type(
                                                             to->cast<TensorType>()->get_shape(), PrimitiveType::f32)
                                                       : PrimitiveType::f32)
                               : stmt->cast_type;

      auto cast_func = [this, cast_op](llvm::Value *value, llvm::Type *type) {
        return this->builder->CreateCast(cast_op, value, type);
      };
      if (!cast_type->is<TensorType>()) {
        llvm_val[stmt] = cast_func(input, tlctx->get_data_type(cast_type));
      } else {
        create_elementwise_cast(stmt, tlctx->get_data_type(cast_type->cast<TensorType>()->get_element_type()),
                                cast_func);
      }

      if (use_f16) {
        auto trunc_func = [this](llvm::Value *value, llvm::Type *type) {
          return this->builder->CreateFPTrunc(value, type);
        };
        auto to_ty = llvm::Type::getHalfTy(*llvm_context);
        if (!cast_type->is<TensorType>()) {
          llvm_val[stmt] = trunc_func(llvm_val[stmt], to_ty);
        } else {
          create_elementwise_cast(stmt, to_ty, trunc_func, /*on_self=*/true);
        }
      }
    } else if (is_real(from.get_element_type()) && is_real(to.get_element_type())) {
      auto t1 = from->is<TensorType>() ? from->cast<TensorType>()->get_element_type() : from.operator->();
      auto t2 = to->is<TensorType>() ? to->cast<TensorType>()->get_element_type() : to.operator->();
      if (data_type_size(t1) < data_type_size(t2)) {
        auto cast_func = [this](llvm::Value *value, llvm::Type *type) {
          return this->builder->CreateFPExt(value, type);
        };
        if (!stmt->cast_type->is<TensorType>()) {
          llvm_val[stmt] = cast_func(input, tlctx->get_data_type(stmt->cast_type));
        } else {
          create_elementwise_cast(stmt, tlctx->get_data_type(stmt->cast_type->cast<TensorType>()->get_element_type()),
                                  cast_func);
        }
      } else {
        if (to->is_primitive(PrimitiveTypeID::f16) ||
            (to->is<TensorType>() && to->cast<TensorType>()->get_element_type()->is_primitive(PrimitiveTypeID::f16))) {
          if (!to->is<TensorType>()) {
            llvm_val[stmt] = builder->CreateFPTrunc(
                builder->CreateFPTrunc(llvm_val[stmt->operand], llvm::Type::getFloatTy(*llvm_context)),
                llvm::Type::getHalfTy(*llvm_context));
          } else {
            auto tensor_type = to->cast<TensorType>();
            llvm::Value *vec = llvm::UndefValue::get(tlctx->get_data_type(to));
            for (int i = 0; i < tensor_type->get_num_elements(); ++i) {
              auto elem = builder->CreateExtractElement(vec, i);
              auto double_trunced =
                  builder->CreateFPTrunc(builder->CreateFPTrunc(elem, llvm::Type::getFloatTy(*llvm_context)),
                                         llvm::Type::getHalfTy(*llvm_context));
              vec = builder->CreateInsertElement(vec, double_trunced, i);
            }
            llvm_val[stmt] = vec;
          }
        } else {
          auto trunc_fn = [this](llvm::Value *value, llvm::Type *type) {
            return this->builder->CreateFPTrunc(value, type);
          };
          auto cast_type = stmt->cast_type->is<TensorType>() ? stmt->cast_type->cast<TensorType>()->get_element_type()
                                                             : stmt->cast_type.operator->();
          if (!stmt->cast_type->is<TensorType>()) {
            llvm_val[stmt] = trunc_fn(input, tlctx->get_data_type(cast_type));
          } else {
            create_elementwise_cast(stmt, tlctx->get_data_type(cast_type->cast<TensorType>()->get_element_type()),
                                    trunc_fn);
          }
        }
      }
    } else if (!is_real(from.get_element_type()) && !is_real(to.get_element_type())) {
      llvm_val[stmt] =
          builder->CreateIntCast(llvm_val[stmt->operand], tlctx->get_data_type(to), is_signed(from.get_element_type()));
    }
  } else if (stmt->op_type == UnaryOpType::cast_bits) {
    QD_ASSERT(data_type_size(stmt->ret_type) == data_type_size(stmt->cast_type));
    if (stmt->operand->ret_type.is_pointer()) {
      QD_ASSERT(is_integral(stmt->cast_type));
      llvm_val[stmt] = builder->CreatePtrToInt(llvm_val[stmt->operand], tlctx->get_data_type(stmt->cast_type));
    } else {
      llvm_val[stmt] = builder->CreateBitCast(llvm_val[stmt->operand], tlctx->get_data_type(stmt->cast_type));
    }
  } else if (op == UnaryOpType::rsqrt) {
    llvm::Function *sqrt_fn =
        llvm::Intrinsic::getOrInsertDeclaration(module.get(), llvm::Intrinsic::sqrt, input->getType());
    auto intermediate = builder->CreateCall(sqrt_fn, input, "sqrt");
    llvm_val[stmt] = builder->CreateFDiv(tlctx->get_constant(stmt->ret_type, 1.0), intermediate);
  } else if (op == UnaryOpType::bit_not) {
    llvm_val[stmt] = builder->CreateNot(input);
  } else if (op == UnaryOpType::neg) {
    if (is_real(stmt->operand->ret_type)) {
      llvm_val[stmt] = builder->CreateFNeg(input, "neg");
    } else {
      llvm_val[stmt] = builder->CreateNeg(input, "neg");
    }
  } else if (op == UnaryOpType::logic_not) {
    llvm_val[stmt] = builder->CreateIsNull(input);
  }
  UNARY_INTRINSIC(round)
  UNARY_INTRINSIC(floor)
  UNARY_INTRINSIC(ceil)
  else {
    emit_extra_unary(stmt);
  }
#undef UNARY_INTRINSIC
}

void TaskCodeGenLLVM::create_elementwise_binary(BinaryOpStmt *stmt,
                                                std::function<llvm::Value *(llvm::Value *lhs, llvm::Value *rhs)> f) {
  QD_ASSERT(stmt->lhs->ret_type->is<TensorType>());
  QD_ASSERT(stmt->rhs->ret_type->is<TensorType>());
  auto lhs_ty = stmt->lhs->ret_type->cast<TensorType>();
  auto rhs_ty = stmt->rhs->ret_type->cast<TensorType>();
  QD_ASSERT(lhs_ty->get_num_elements() == rhs_ty->get_num_elements());
  auto lhs_vec = llvm_val[stmt->lhs];
  auto rhs_vec = llvm_val[stmt->rhs];
  auto elt_type_name = data_type_name(lhs_ty->get_element_type());
  llvm::Value *result = llvm::UndefValue::get(tlctx->get_data_type(stmt->ret_type));
  for (int i = 0; i < lhs_ty->get_num_elements(); ++i) {
    auto lhs = builder->CreateExtractElement(lhs_vec, i);
    auto rhs = builder->CreateExtractElement(rhs_vec, i);
    result = builder->CreateInsertElement(result, f(lhs, rhs), i);
  }
  llvm_val[stmt] = result;
}

void TaskCodeGenLLVM::visit(BinaryOpStmt *stmt) {
  auto op = stmt->op_type;
  auto ret_type = stmt->ret_type;

  if (op == BinaryOpType::add) {
    if (is_real(stmt->ret_type.get_element_type())) {
      llvm_val[stmt] = builder->CreateFAdd(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
#if defined(__clang__) || defined(__GNUC__)
    } else if (compile_config.debug && is_integral(stmt->ret_type)) {
      llvm_val[stmt] = call("debug_add_" + stmt->ret_type->to_string(), get_arg(0), llvm_val[stmt->lhs],
                            llvm_val[stmt->rhs], builder->CreateGlobalStringPtr(stmt->get_tb()));
#endif
    } else {
      llvm_val[stmt] = builder->CreateAdd(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::sub) {
    if (is_real(stmt->ret_type.get_element_type())) {
      llvm_val[stmt] = builder->CreateFSub(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
#if defined(__clang__) || defined(__GNUC__)
    } else if (compile_config.debug && is_integral(stmt->ret_type)) {
      llvm_val[stmt] = call("debug_sub_" + stmt->ret_type->to_string(), get_arg(0), llvm_val[stmt->lhs],
                            llvm_val[stmt->rhs], builder->CreateGlobalStringPtr(stmt->get_tb()));
#endif
    } else {
      llvm_val[stmt] = builder->CreateSub(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::mul) {
    if (is_real(stmt->ret_type.get_element_type())) {
      llvm_val[stmt] = builder->CreateFMul(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
#if defined(__clang__) || defined(__GNUC__)
    } else if (compile_config.debug && is_integral(stmt->ret_type)) {
      llvm_val[stmt] = call("debug_mul_" + stmt->ret_type->to_string(), get_arg(0), llvm_val[stmt->lhs],
                            llvm_val[stmt->rhs], builder->CreateGlobalStringPtr(stmt->get_tb()));
#endif
    } else {
      llvm_val[stmt] = builder->CreateMul(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::div) {
    if (is_real(stmt->ret_type.get_element_type())) {
      llvm_val[stmt] = builder->CreateFDiv(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    } else if (is_signed(stmt->ret_type.get_element_type())) {
      llvm_val[stmt] = builder->CreateSDiv(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    } else {
      llvm_val[stmt] = builder->CreateUDiv(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::mod) {
    llvm_val[stmt] = builder->CreateSRem(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
  } else if (op == BinaryOpType::logical_and) {
    auto *lhs = builder->CreateIsNotNull(llvm_val[stmt->lhs]);
    auto *rhs = builder->CreateIsNotNull(llvm_val[stmt->rhs]);
    llvm_val[stmt] = builder->CreateAnd(lhs, rhs);
    llvm_val[stmt] = builder->CreateZExtOrTrunc(llvm_val[stmt], tlctx->get_data_type(stmt->ret_type));
  } else if (op == BinaryOpType::logical_or) {
    auto *lhs = builder->CreateIsNotNull(llvm_val[stmt->lhs]);
    auto *rhs = builder->CreateIsNotNull(llvm_val[stmt->rhs]);
    llvm_val[stmt] = builder->CreateOr(lhs, rhs);
    llvm_val[stmt] = builder->CreateZExtOrTrunc(llvm_val[stmt], tlctx->get_data_type(stmt->ret_type));
  } else if (op == BinaryOpType::bit_and) {
    llvm_val[stmt] = builder->CreateAnd(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
  } else if (op == BinaryOpType::bit_or) {
    llvm_val[stmt] = builder->CreateOr(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
  } else if (op == BinaryOpType::bit_xor) {
    llvm_val[stmt] = builder->CreateXor(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
  } else if (op == BinaryOpType::bit_shl) {
#if defined(__clang__) || defined(__GNUC__)
    if (compile_config.debug && is_integral(stmt->ret_type)) {
      llvm_val[stmt] = call("debug_shl_" + stmt->ret_type->to_string(), get_arg(0), llvm_val[stmt->lhs],
                            llvm_val[stmt->rhs], builder->CreateGlobalStringPtr(stmt->get_tb()));
    } else {
      llvm_val[stmt] = builder->CreateShl(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
#else
    llvm_val[stmt] = builder->CreateShl(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
#endif
  } else if (op == BinaryOpType::bit_sar) {
    if (is_signed(stmt->lhs->ret_type.get_element_type())) {
      llvm_val[stmt] = builder->CreateAShr(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    } else {
      llvm_val[stmt] = builder->CreateLShr(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::max) {
#define BINARYOP_MAX(x)                                                         \
  else if (ret_type->is_primitive(PrimitiveTypeID::x)) {                        \
    llvm_val[stmt] = call("max_" #x, llvm_val[stmt->lhs], llvm_val[stmt->rhs]); \
  }

    if (is_real(ret_type.get_element_type())) {
      llvm_val[stmt] = builder->CreateMaxNum(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
    BINARYOP_MAX(u16)
    BINARYOP_MAX(i16)
    BINARYOP_MAX(u32)
    BINARYOP_MAX(i32)
    BINARYOP_MAX(u64)
    BINARYOP_MAX(i64)
    else {
      if (auto tensor_ty = ret_type->cast<TensorType>()) {
        auto elt_ty = tensor_ty->get_element_type();
        QD_ASSERT(elt_ty->is_primitive(PrimitiveTypeID::u16) || elt_ty->is_primitive(PrimitiveTypeID::i16) ||
                  elt_ty->is_primitive(PrimitiveTypeID::u32) || elt_ty->is_primitive(PrimitiveTypeID::i32) ||
                  elt_ty->is_primitive(PrimitiveTypeID::u64) || elt_ty->is_primitive(PrimitiveTypeID::i64));
        auto dtype_name = data_type_name(elt_ty);
        auto binary_max = [this, &dtype_name](llvm::Value *lhs, llvm::Value *rhs) {
          return call("max_" + dtype_name, lhs, rhs);
        };
        create_elementwise_binary(stmt, binary_max);
      } else {
        QD_P(data_type_name(ret_type));
        QD_NOT_IMPLEMENTED
      }
    }
  } else if (op == BinaryOpType::min) {
#define BINARYOP_MIN(x)                                                         \
  else if (ret_type->is_primitive(PrimitiveTypeID::x)) {                        \
    llvm_val[stmt] = call("min_" #x, llvm_val[stmt->lhs], llvm_val[stmt->rhs]); \
  }

    if (is_real(ret_type.get_element_type())) {
      llvm_val[stmt] = builder->CreateMinNum(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
    BINARYOP_MIN(u16)
    BINARYOP_MIN(i16)
    BINARYOP_MIN(u32)
    BINARYOP_MIN(i32)
    BINARYOP_MIN(u64)
    BINARYOP_MIN(i64)
    else {
      if (auto tensor_ty = ret_type->cast<TensorType>()) {
        auto elt_ty = tensor_ty->get_element_type();
        QD_ASSERT(elt_ty->is_primitive(PrimitiveTypeID::u16) || elt_ty->is_primitive(PrimitiveTypeID::i16) ||
                  elt_ty->is_primitive(PrimitiveTypeID::u32) || elt_ty->is_primitive(PrimitiveTypeID::i32) ||
                  elt_ty->is_primitive(PrimitiveTypeID::u64) || elt_ty->is_primitive(PrimitiveTypeID::i64));
        auto dtype_name = data_type_name(elt_ty);
        auto binary_min = [this, &dtype_name](llvm::Value *lhs, llvm::Value *rhs) {
          return call("min_" + dtype_name, lhs, rhs);
        };
        create_elementwise_binary(stmt, binary_min);
      } else {
        QD_P(data_type_name(ret_type));
        QD_NOT_IMPLEMENTED
      }
    }
  } else if (is_comparison(op)) {
    llvm::Value *cmp = nullptr;
    auto input_type = stmt->lhs->ret_type;
    if (op == BinaryOpType::cmp_eq) {
      if (is_real(input_type.get_element_type())) {
        cmp = builder->CreateFCmpOEQ(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        cmp = builder->CreateICmpEQ(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      }
    } else if (op == BinaryOpType::cmp_le) {
      if (is_real(input_type.get_element_type())) {
        cmp = builder->CreateFCmpOLE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        if (is_signed(input_type.get_element_type())) {
          cmp = builder->CreateICmpSLE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        } else {
          cmp = builder->CreateICmpULE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        }
      }
    } else if (op == BinaryOpType::cmp_ge) {
      if (is_real(input_type.get_element_type())) {
        cmp = builder->CreateFCmpOGE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        if (is_signed(input_type.get_element_type())) {
          cmp = builder->CreateICmpSGE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        } else {
          cmp = builder->CreateICmpUGE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        }
      }
    } else if (op == BinaryOpType::cmp_lt) {
      if (is_real(input_type.get_element_type())) {
        cmp = builder->CreateFCmpOLT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        if (is_signed(input_type.get_element_type())) {
          cmp = builder->CreateICmpSLT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        } else {
          cmp = builder->CreateICmpULT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        }
      }
    } else if (op == BinaryOpType::cmp_gt) {
      if (is_real(input_type.get_element_type())) {
        cmp = builder->CreateFCmpOGT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        if (is_signed(input_type.get_element_type())) {
          cmp = builder->CreateICmpSGT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        } else {
          cmp = builder->CreateICmpUGT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        }
      }
    } else if (op == BinaryOpType::cmp_ne) {
      if (is_real(input_type.get_element_type())) {
        cmp = builder->CreateFCmpONE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        cmp = builder->CreateICmpNE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      }
    } else {
      QD_NOT_IMPLEMENTED
    }
    llvm_val[stmt] = cmp;
  } else {
    // This branch contains atan2 and pow which use runtime.cpp function for
    // **real** type. We don't have f16 support there so promoting to f32 is
    // necessary.
    llvm::Value *lhs = llvm_val[stmt->lhs];
    llvm::Value *rhs = llvm_val[stmt->rhs];
    if (stmt->lhs->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      lhs = builder->CreateFPExt(lhs, llvm::Type::getFloatTy(*llvm_context));
    }
    if (stmt->rhs->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      rhs = builder->CreateFPExt(rhs, llvm::Type::getFloatTy(*llvm_context));
    }
    if (ret_type->is_primitive(PrimitiveTypeID::f16)) {
      ret_type = PrimitiveType::f32;
    }

    if (op == BinaryOpType::atan2) {
      if (arch_is_cpu(current_arch())) {
        if (ret_type->is_primitive(PrimitiveTypeID::f32)) {
          llvm_val[stmt] = call("atan2_f32", lhs, rhs);
        } else if (ret_type->is_primitive(PrimitiveTypeID::f64)) {
          llvm_val[stmt] = call("atan2_f64", lhs, rhs);
        } else {
          QD_P(data_type_name(ret_type));
          QD_NOT_IMPLEMENTED
        }
      } else {
        QD_NOT_IMPLEMENTED
      }
    } else if (op == BinaryOpType::pow) {
      if (arch_is_cpu(current_arch())) {
        // Note that ret_type here cannot be integral because pow with an
        // integral exponent has been demoted in the demote_operations pass
        if (ret_type->is_primitive(PrimitiveTypeID::f32)) {
          llvm_val[stmt] = call("pow_f32", lhs, rhs);
        } else if (ret_type->is_primitive(PrimitiveTypeID::f64)) {
          llvm_val[stmt] = call("pow_f64", lhs, rhs);
        } else {
          QD_P(data_type_name(ret_type));
          QD_NOT_IMPLEMENTED
        }
      } else {
        QD_NOT_IMPLEMENTED
      }
    } else {
      QD_P(binary_op_type_name(op));
      QD_NOT_IMPLEMENTED
    }

    // Convert back to f16 if applicable.
    if (stmt->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      llvm_val[stmt] = builder->CreateFPTrunc(llvm_val[stmt], llvm::Type::getHalfTy(*llvm_context));
    }
  }
}

void TaskCodeGenLLVM::visit(TernaryOpStmt *stmt) {
  QD_ASSERT(stmt->op_type == TernaryOpType::select);
  llvm_val[stmt] =
      builder->CreateSelect(builder->CreateIsNotNull(llvm_val[stmt->op1]), llvm_val[stmt->op2], llvm_val[stmt->op3]);
}

void TaskCodeGenLLVM::visit(IfStmt *if_stmt) {
  // TODO: take care of vectorized cases
  llvm::BasicBlock *true_block = llvm::BasicBlock::Create(*llvm_context, "true_block", func);
  llvm::BasicBlock *false_block = llvm::BasicBlock::Create(*llvm_context, "false_block", func);
  llvm::BasicBlock *after_if = llvm::BasicBlock::Create(*llvm_context, "after_if", func);
  llvm::Value *cond = builder->CreateIsNotNull(llvm_val[if_stmt->cond]);
  builder->CreateCondBr(cond, true_block, false_block);
  builder->SetInsertPoint(true_block);
  if (if_stmt->true_statements) {
    if_stmt->true_statements->accept(this);
  }
  if (!returned) {
    builder->CreateBr(after_if);
  } else {
    returned = false;
  }
  builder->SetInsertPoint(false_block);
  if (if_stmt->false_statements) {
    if_stmt->false_statements->accept(this);
  }
  if (!returned) {
    builder->CreateBr(after_if);
  } else {
    returned = false;
  }
  builder->SetInsertPoint(after_if);
}

llvm::Value *TaskCodeGenLLVM::create_print(std::string tag, DataType dt, llvm::Value *value) {
  if (!arch_is_cpu(compile_config.arch)) {
    QD_WARN("print not supported on arch {}", arch_name(compile_config.arch));
    return nullptr;
  }
  std::vector<llvm::Value *> args;
  std::string format = data_type_format(dt);
  auto runtime_printf = call("LLVMRuntime_get_host_printf", get_runtime());
  args.push_back(
      builder->CreateGlobalStringPtr(("[llvm codegen debug] " + tag + " = " + format + "\n").c_str(), "format_string"));
  if (dt->is_primitive(PrimitiveTypeID::f32))
    value = builder->CreateFPExt(value, tlctx->get_data_type(PrimitiveType::f64));
  args.push_back(value);

  auto func_type_func = get_runtime_function("get_func_type_host_printf");
  return call(runtime_printf, func_type_func->getFunctionType(), std::move(args));
}

llvm::Value *TaskCodeGenLLVM::create_print(std::string tag, llvm::Value *value) {
  if (value->getType() == llvm::Type::getFloatTy(*llvm_context))
    return create_print(tag, TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::f32), value);
  else if (value->getType() == llvm::Type::getInt32Ty(*llvm_context))
    return create_print(tag, TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i32), value);
  else if (value->getType() == llvm::Type::getHalfTy(*llvm_context)) {
    auto extended = builder->CreateFPExt(value, llvm::Type::getFloatTy(*llvm_context));
    return create_print(tag, TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::f32), extended);
  } else if (value->getType() == llvm::Type::getInt64Ty(*llvm_context))
    return create_print(tag, TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i64), value);
  else if (value->getType() == llvm::Type::getInt16Ty(*llvm_context))
    return create_print(tag, TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i16), value);
  else
    QD_NOT_IMPLEMENTED
}

void TaskCodeGenLLVM::visit(PrintStmt *stmt) {
  std::vector<llvm::Value *> args;
  std::string formats;
  auto value_for_printf = [this](llvm::Value *to_print, DataType dtype) {
    if (dtype->is_primitive(PrimitiveTypeID::f32) || dtype->is_primitive(PrimitiveTypeID::f16))
      return this->builder->CreateFPExt(to_print, this->tlctx->get_data_type(PrimitiveType::f64));
    if (dtype->is_primitive(PrimitiveTypeID::i8))
      return builder->CreateSExt(to_print, tlctx->get_data_type(PrimitiveType::i16));
    if (dtype->is_primitive(PrimitiveTypeID::u8))
      return builder->CreateZExt(to_print, tlctx->get_data_type(PrimitiveType::u16));
    if (dtype->is_primitive(PrimitiveTypeID::u1))
      return builder->CreateZExt(to_print, tlctx->get_data_type(PrimitiveType::i32));
    return to_print;
  };
  for (auto i = 0; i < stmt->contents.size(); ++i) {
    auto const &content = stmt->contents[i];
    auto const &format = stmt->formats[i];

    if (std::holds_alternative<Stmt *>(content)) {
      auto arg_stmt = std::get<Stmt *>(content);
      auto value = llvm_val[arg_stmt];
      auto value_type = value->getType();
      if (arg_stmt->ret_type->is<TensorType>()) {
        auto dtype = arg_stmt->ret_type->cast<TensorType>();
        auto elem_type = dtype->get_element_type();
        for (int i = 0; i < dtype->get_num_elements(); ++i) {
          if (codegen_vector_type(compile_config)) {
            QD_ASSERT(llvm::dyn_cast<llvm::VectorType>(value_type));
            auto elem = builder->CreateExtractElement(value, i);
            args.push_back(value_for_printf(elem, elem_type));
          } else {
            QD_ASSERT(llvm::dyn_cast<llvm::ArrayType>(value_type));
            auto elem = builder->CreateExtractValue(value, i);
            args.push_back(value_for_printf(elem, elem_type));
          }
        }
        formats += data_type_format(arg_stmt->ret_type);
      } else {
        args.push_back(value_for_printf(value, arg_stmt->ret_type));
        formats += merge_printf_specifier(format, data_type_format(arg_stmt->ret_type));
      }
    } else {
      auto arg_str = std::get<std::string>(content);
      auto value = builder->CreateGlobalStringPtr(arg_str, "content_string");
      args.push_back(value);
      formats += "%s";
    }
  }
  auto runtime_printf = call("LLVMRuntime_get_host_printf", get_runtime());
  args.insert(args.begin(), builder->CreateGlobalStringPtr(formats.c_str(), "format_string"));
  auto func_type_func = get_runtime_function("get_func_type_host_printf");
  llvm_val[stmt] = call(runtime_printf, func_type_func->getFunctionType(), std::move(args));
}

void TaskCodeGenLLVM::visit(ConstStmt *stmt) {
  auto val = stmt->val;
  if (val.dt->is_primitive(PrimitiveTypeID::f32)) {
    llvm_val[stmt] = llvm::ConstantFP::get(*llvm_context, llvm::APFloat(val.val_float32()));
  } else if (val.dt->is_primitive(PrimitiveTypeID::f16)) {
    llvm_val[stmt] = llvm::ConstantFP::get(llvm::Type::getHalfTy(*llvm_context), val.val_float16());
  } else if (val.dt->is_primitive(PrimitiveTypeID::f64)) {
    llvm_val[stmt] = llvm::ConstantFP::get(*llvm_context, llvm::APFloat(val.val_float64()));
  } else if (val.dt->is_primitive(PrimitiveTypeID::u1)) {
    llvm_val[stmt] = llvm::ConstantInt::get(*llvm_context, llvm::APInt(1, (uint64)val.val_uint1(), false));
  } else if (val.dt->is_primitive(PrimitiveTypeID::i8)) {
    llvm_val[stmt] = llvm::ConstantInt::get(*llvm_context, llvm::APInt(8, (uint64)val.val_int8(), true));
  } else if (val.dt->is_primitive(PrimitiveTypeID::u8)) {
    llvm_val[stmt] = llvm::ConstantInt::get(*llvm_context, llvm::APInt(8, (uint64)val.val_uint8(), false));
  } else if (val.dt->is_primitive(PrimitiveTypeID::i16)) {
    llvm_val[stmt] = llvm::ConstantInt::get(*llvm_context, llvm::APInt(16, (uint64)val.val_int16(), true));
  } else if (val.dt->is_primitive(PrimitiveTypeID::u16)) {
    llvm_val[stmt] = llvm::ConstantInt::get(*llvm_context, llvm::APInt(16, (uint64)val.val_uint16(), false));
  } else if (val.dt->is_primitive(PrimitiveTypeID::i32)) {
    llvm_val[stmt] = llvm::ConstantInt::get(*llvm_context, llvm::APInt(32, (uint64)val.val_int32(), true));
  } else if (val.dt->is_primitive(PrimitiveTypeID::u32)) {
    llvm_val[stmt] = llvm::ConstantInt::get(*llvm_context, llvm::APInt(32, (uint64)val.val_uint32(), false));
  } else if (val.dt->is_primitive(PrimitiveTypeID::i64)) {
    llvm_val[stmt] = llvm::ConstantInt::get(*llvm_context, llvm::APInt(64, (uint64)val.val_int64(), true));
  } else if (val.dt->is_primitive(PrimitiveTypeID::u64)) {
    llvm_val[stmt] = llvm::ConstantInt::get(*llvm_context, llvm::APInt(64, val.val_uint64(), false));
  } else {
    QD_P(data_type_name(val.dt));
    QD_NOT_IMPLEMENTED;
  }
}

void TaskCodeGenLLVM::visit(WhileControlStmt *stmt) {
  using namespace llvm;

  BasicBlock *after_break = BasicBlock::Create(*llvm_context, "after_break", func);
  QD_ASSERT(current_while_after_loop);
  auto *cond = builder->CreateIsNull(llvm_val[stmt->cond]);
  builder->CreateCondBr(cond, current_while_after_loop, after_break);
  builder->SetInsertPoint(after_break);
}

void TaskCodeGenLLVM::visit(ContinueStmt *stmt) {
  using namespace llvm;
  auto stmt_in_off_range_for = [stmt]() {
    QD_ASSERT(stmt->scope != nullptr);
    if (auto *offl = stmt->scope->cast<OffloadedStmt>(); offl) {
      QD_ASSERT(offl->task_type == OffloadedStmt::TaskType::range_for ||
                offl->task_type == OffloadedStmt::TaskType::struct_for);
      return offl->task_type == OffloadedStmt::TaskType::range_for;
    }
    return false;
  };
  if (stmt_in_off_range_for()) {
    builder->CreateRetVoid();
  } else {
    QD_ASSERT(current_loop_reentry != nullptr);
    builder->CreateBr(current_loop_reentry);
  }
  // Stmts after continue are useless, so we switch the insertion point to
  // /dev/null. In LLVM IR, the "after_continue" label shows "No predecessors!".
  BasicBlock *after_continue = BasicBlock::Create(*llvm_context, "after_continue", func);
  builder->SetInsertPoint(after_continue);
}

void TaskCodeGenLLVM::visit(WhileStmt *stmt) {
  using namespace llvm;
  BasicBlock *body = BasicBlock::Create(*llvm_context, "while_loop_body", func);
  builder->CreateBr(body);
  builder->SetInsertPoint(body);
  auto lrg = make_loop_reentry_guard(this);
  current_loop_reentry = body;

  BasicBlock *after_loop = BasicBlock::Create(*llvm_context, "after_while", func);
  auto walg = make_while_after_loop_guard(this);
  current_while_after_loop = after_loop;

  stmt->body->accept(this);

  if (!returned) {
    builder->CreateBr(body);  // jump to head
  } else {
    returned = false;
  }

  builder->SetInsertPoint(after_loop);
}

llvm::Value *TaskCodeGenLLVM::cast_pointer(llvm::Value *val, std::string dest_ty_name, int addr_space) {
  return builder->CreateBitCast(val, llvm::PointerType::get(get_runtime_type(dest_ty_name), addr_space));
}

void TaskCodeGenLLVM::emit_list_gen(OffloadedStmt *listgen) {
  auto snode_child = listgen->snode;
  auto snode_parent = listgen->snode->parent;
  auto meta_child = cast_pointer(emit_struct_meta(snode_child), "StructMeta");
  auto meta_parent = cast_pointer(emit_struct_meta(snode_parent), "StructMeta");
  if (snode_parent->type == SNodeType::root) {
    // Since there's only one container to expand, we need a special kernel for
    // more parallelism.
    call("element_listgen_root", get_runtime(), meta_parent, meta_child);
  } else {
    call("element_listgen_nonroot", get_runtime(), meta_parent, meta_child);
  }
}

void TaskCodeGenLLVM::emit_gc(OffloadedStmt *stmt) {
  auto snode = stmt->snode->id;
  call("node_gc", get_runtime(), tlctx->get_constant(snode));
}

void TaskCodeGenLLVM::create_increment(llvm::Value *ptr, llvm::Value *value) {
  auto original_value = builder->CreateLoad(value->getType(), ptr);
  builder->CreateStore(builder->CreateAdd(original_value, value), ptr);
}

void TaskCodeGenLLVM::create_naive_range_for(RangeForStmt *for_stmt) {
  using namespace llvm;
  BasicBlock *body = BasicBlock::Create(*llvm_context, "for_loop_body", func);
  BasicBlock *loop_inc = BasicBlock::Create(*llvm_context, "for_loop_inc", func);
  BasicBlock *after_loop = BasicBlock::Create(*llvm_context, "after_for", func);
  BasicBlock *loop_test = BasicBlock::Create(*llvm_context, "for_loop_test", func);

  auto loop_var_ty = tlctx->get_data_type(PrimitiveType::i32);

  auto loop_var = create_entry_block_alloca(PrimitiveType::i32);
  loop_vars_llvm[for_stmt].push_back(loop_var);

  if (!for_stmt->reversed) {
    builder->CreateStore(llvm_val[for_stmt->begin], loop_var);
  } else {
    builder->CreateStore(builder->CreateSub(llvm_val[for_stmt->end], tlctx->get_constant(1)), loop_var);
  }
  builder->CreateBr(loop_test);

  {
    // test block
    builder->SetInsertPoint(loop_test);
    llvm::Value *cond;
    if (!for_stmt->reversed) {
      cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT, builder->CreateLoad(loop_var_ty, loop_var),
                                 llvm_val[for_stmt->end]);
    } else {
      cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SGE, builder->CreateLoad(loop_var_ty, loop_var),
                                 llvm_val[for_stmt->begin]);
    }
    builder->CreateCondBr(cond, body, after_loop);
  }

  {
    {
      auto lrg = make_loop_reentry_guard(this);
      // The continue stmt should jump to the loop-increment block!
      current_loop_reentry = loop_inc;
      // body cfg
      builder->SetInsertPoint(body);

      for_stmt->body->accept(this);
    }
    if (!returned) {
      builder->CreateBr(loop_inc);
    } else {
      returned = false;
    }
    builder->SetInsertPoint(loop_inc);

    if (!for_stmt->reversed) {
      create_increment(loop_var, tlctx->get_constant(1));
    } else {
      create_increment(loop_var, tlctx->get_constant(-1));
    }
    builder->CreateBr(loop_test);
  }

  // next cfg
  builder->SetInsertPoint(after_loop);
}

void TaskCodeGenLLVM::visit(RangeForStmt *for_stmt) {
  create_naive_range_for(for_stmt);
}

llvm::Value *TaskCodeGenLLVM::bitcast_from_u64(llvm::Value *val, DataType type) {
  llvm::Type *dest_ty = nullptr;
  QD_ASSERT(!type->is<PointerType>());
  if (auto qit = type->cast<QuantIntType>()) {
    if (qit->get_is_signed())
      dest_ty = tlctx->get_data_type(PrimitiveType::i32);
    else
      dest_ty = tlctx->get_data_type(PrimitiveType::u32);
  } else {
    dest_ty = tlctx->get_data_type(type);
  }
  auto dest_bits = dest_ty->getPrimitiveSizeInBits();
  if (dest_ty == llvm::Type::getHalfTy(*llvm_context)) {
    // if dest_ty == half, CreateTrunc will only keep low 16bits of mantissa
    // which doesn't mean anything.
    // So we truncate to 32 bits first and then fptrunc to half if applicable
    auto truncated = builder->CreateTrunc(val, llvm::Type::getIntNTy(*llvm_context, 32));
    auto casted = builder->CreateBitCast(truncated, llvm::Type::getFloatTy(*llvm_context));
    return builder->CreateFPTrunc(casted, llvm::Type::getHalfTy(*llvm_context));
  } else {
    auto truncated = builder->CreateTrunc(val, llvm::Type::getIntNTy(*llvm_context, dest_bits));

    return builder->CreateBitCast(truncated, dest_ty);
  }
}

llvm::Value *TaskCodeGenLLVM::bitcast_to_u64(llvm::Value *val, DataType type) {
  auto intermediate_bits = 0;
  if (type.is_pointer()) {
    return builder->CreatePtrToInt(val, tlctx->get_data_type<int64>());
  }
  if (auto qit = type->cast<QuantIntType>()) {
    intermediate_bits = data_type_bits(qit->get_compute_type());
  } else {
    intermediate_bits = tlctx->get_data_type(type)->getPrimitiveSizeInBits();
  }
  llvm::Type *dest_ty = tlctx->get_data_type<int64>();
  llvm::Type *intermediate_type = nullptr;
  if (val->getType() == llvm::Type::getHalfTy(*llvm_context)) {
    val = builder->CreateFPExt(val, tlctx->get_data_type<float>());
    intermediate_type = tlctx->get_data_type<int32>();
  } else {
    intermediate_type = llvm::Type::getIntNTy(*llvm_context, intermediate_bits);
  }
  return builder->CreateZExt(builder->CreateBitCast(val, intermediate_type), dest_ty);
}

void TaskCodeGenLLVM::visit(ArgLoadStmt *stmt) {
  llvm_val[stmt] = get_struct_arg(stmt->arg_id, stmt->create_load);
}

void TaskCodeGenLLVM::visit(ReturnStmt *stmt) {
  auto types = stmt->element_types();
  if (std::any_of(types.begin(), types.end(), [](const DataType &t) { return t.is_pointer(); })) {
    QD_NOT_IMPLEMENTED
  } else {
    QD_ASSERT(stmt->values.size() == current_callable->ret_type->get_flattened_num_elements());
    auto *buffer = call("RuntimeContext_get_result_buffer", get_context());
    set_struct_to_buffer(current_callable->ret_type, buffer, stmt->values);
  }
  builder->CreateBr(final_block);
  returned = true;
}

void TaskCodeGenLLVM::visit(LocalLoadStmt *stmt) {
  // FIXME: get ptr_ty from quadrants instead of llvm.
  llvm::Type *ptr_ty = nullptr;
  auto *val = llvm_val[stmt->src];
  if (auto *alloc = llvm::dyn_cast<llvm::AllocaInst>(val))
    ptr_ty = alloc->getAllocatedType();
  if (!ptr_ty && stmt->src->element_type().is_pointer()) {
    ptr_ty = tlctx->get_data_type(stmt->src->element_type().ptr_removed());
  }
  QD_ASSERT(ptr_ty);
  llvm_val[stmt] = builder->CreateLoad(ptr_ty, llvm_val[stmt->src]);
}

void TaskCodeGenLLVM::visit(LocalStoreStmt *stmt) {
  builder->CreateStore(llvm_val[stmt->val], llvm_val[stmt->dest]);
}

void TaskCodeGenLLVM::visit(AssertStmt *stmt) {
  QD_ASSERT((int)stmt->args.size() <= quadrants_error_message_max_num_arguments);

  // Emit the test inline and keep argument marshalling plus the runtime call in a cold block reached only on
  // failure. An unconditional runtime call at every site puts an ABI call (with its argument buffer and stores) on
  // the hot path of every checked access; on GPU the resulting register spills are backed by a per-resident-thread
  // local-memory reservation across the whole device, which multiplies to gigabytes for bounds-checked kernels.
  //
  // On CPU, use the context-aware variant that returns non-zero on failure so we can emit an early return and
  // avoid the subsequent out-of-bounds memory access. On GPU, asm("exit;") kills the thread directly when asserts
  // are enabled at runtime; otherwise the cold block falls through to the access, as before.
  bool use_ctx_variant = arch_is_cpu(current_arch());
  auto *test = builder->CreateIsNotNull(llvm_val[stmt->cond]);

  auto *assert_fail = llvm::BasicBlock::Create(*llvm_context, "assert_fail", func);
  auto *assert_cont = llvm::BasicBlock::Create(*llvm_context, "assert_cont", func);
  builder->CreateCondBr(test, assert_cont, assert_fail);
  builder->SetInsertPoint(assert_fail);

  std::vector<llvm::Value *> args;
  args.emplace_back(use_ctx_variant ? get_context() : get_runtime());

  constexpr int max_args_by_value = 8;
  llvm::Value *result = nullptr;
  if ((int)stmt->args.size() <= max_args_by_value) {
    // Fast path: pass the arguments by value, so that no per-site argument buffer is ever allocated.
    args.emplace_back(builder->CreateGlobalStringPtr(stmt->text));
    args.emplace_back(tlctx->get_constant((int)stmt->args.size()));
    for (int i = 0; i < max_args_by_value; i++) {
      if (i < (int)stmt->args.size()) {
        auto arg = stmt->args[i];
        QD_ASSERT(llvm_val[arg]);
        auto cast_type = llvm::Type::getIntNTy(*llvm_context, 8 * (std::size_t)data_type_size(arg->ret_type));
        auto cast_int = builder->CreateBitCast(llvm_val[arg], cast_type);
        args.emplace_back(builder->CreateZExt(cast_int, llvm::Type::getInt64Ty(*llvm_context)));
      } else {
        args.emplace_back(tlctx->get_constant((uint64)0));
      }
    }
    result =
        call(use_ctx_variant ? "quadrants_assert_format_ctx_args8" : "quadrants_assert_format_args8", std::move(args));
  } else {
    auto argument_buffer_size = llvm::ArrayType::get(llvm::Type::getInt64Ty(*llvm_context), stmt->args.size());
    auto arguments = create_entry_block_alloca(argument_buffer_size);

    args.emplace_back(test);
    args.emplace_back(builder->CreateGlobalStringPtr(stmt->text));

    for (int i = 0; i < stmt->args.size(); i++) {
      auto arg = stmt->args[i];
      QD_ASSERT(llvm_val[arg]);

      // First convert the argument to an integral type with the same number of bits:
      auto cast_type = llvm::Type::getIntNTy(*llvm_context, 8 * (std::size_t)data_type_size(arg->ret_type));
      auto cast_int = builder->CreateBitCast(llvm_val[arg], cast_type);

      // Then zero-extend the conversion result into int64:
      auto cast_int64 = builder->CreateZExt(cast_int, llvm::Type::getInt64Ty(*llvm_context));

      // Finally store the int64 value to the argument buffer:
      builder->CreateStore(cast_int64, builder->CreateGEP(argument_buffer_size, arguments,
                                                          {tlctx->get_constant(0), tlctx->get_constant(i)}));
    }

    args.emplace_back(tlctx->get_constant((int)stmt->args.size()));
    args.emplace_back(
        builder->CreateGEP(argument_buffer_size, arguments, {tlctx->get_constant(0), tlctx->get_constant(0)}));

    result = call(use_ctx_variant ? "quadrants_assert_format_ctx" : "quadrants_assert_format", std::move(args));
  }
  llvm_val[stmt] = result;

  if (use_ctx_variant) {
    auto *assert_abort = llvm::BasicBlock::Create(*llvm_context, "assert_abort", func);
    auto *failed = builder->CreateICmpNE(result, tlctx->get_constant(0));
    builder->CreateCondBr(failed, assert_abort, assert_cont);
    builder->SetInsertPoint(assert_abort);
    builder->CreateRetVoid();
  } else {
    builder->CreateBr(assert_cont);
  }
  builder->SetInsertPoint(assert_cont);
}

void TaskCodeGenLLVM::visit(SNodeOpStmt *stmt) {
  auto snode = stmt->snode;
  if (stmt->op_type == SNodeOpType::allocate) {
    QD_ASSERT(snode->type == SNodeType::dynamic);
    QD_ASSERT(stmt->ret_type.is_pointer() && stmt->ret_type.ptr_removed()->is_primitive(PrimitiveTypeID::gen));
    auto ptr = call(snode, llvm_val[stmt->ptr], "allocate", {llvm_val[stmt->val]});
    llvm_val[stmt] = ptr;
  } else if (stmt->op_type == SNodeOpType::length) {
    QD_ASSERT(snode->type == SNodeType::dynamic);
    llvm_val[stmt] = call(snode, llvm_val[stmt->ptr], "get_num_elements", {});
  } else if (stmt->op_type == SNodeOpType::is_active) {
    llvm_val[stmt] = call(snode, llvm_val[stmt->ptr], "is_active", {llvm_val[stmt->val]});
  } else if (stmt->op_type == SNodeOpType::activate) {
    llvm_val[stmt] = call(snode, llvm_val[stmt->ptr], "activate", {llvm_val[stmt->val]});
  } else if (stmt->op_type == SNodeOpType::deactivate) {
    if (snode->type == SNodeType::pointer || snode->type == SNodeType::hash || snode->type == SNodeType::bitmasked) {
      llvm_val[stmt] = call(snode, llvm_val[stmt->ptr], "deactivate", {llvm_val[stmt->val]});
    } else if (snode->type == SNodeType::dynamic) {
      llvm_val[stmt] = call(snode, llvm_val[stmt->ptr], "deactivate", {});
    }
  } else {
    QD_NOT_IMPLEMENTED
  }
}

llvm::Value *TaskCodeGenLLVM::optimized_reduction(AtomicOpStmt *stmt) {
  return nullptr;
}

llvm::Value *TaskCodeGenLLVM::quant_type_atomic(AtomicOpStmt *stmt) {
  // TODO(type): support all AtomicOpTypes on quant types
  if (stmt->op_type != AtomicOpType::add) {
    return nullptr;
  }

  auto dst_type = stmt->dest->ret_type->as<PointerType>()->get_pointee_type();
  if (auto qit = dst_type->cast<QuantIntType>()) {
    return atomic_add_quant_int(llvm_val[stmt->dest],
                                tlctx->get_data_type(stmt->dest->as<GetChStmt>()->input_snode->physical_type), qit,
                                llvm_val[stmt->val], is_signed(stmt->val->ret_type));
  } else if (auto qfxt = dst_type->cast<QuantFixedType>()) {
    return atomic_add_quant_fixed(llvm_val[stmt->dest],
                                  tlctx->get_data_type(stmt->dest->as<GetChStmt>()->input_snode->physical_type), qfxt,
                                  llvm_val[stmt->val]);
  } else {
    return nullptr;
  }
}

llvm::Value *TaskCodeGenLLVM::integral_type_atomic(AtomicOpStmt *stmt) {
  if (!is_integral(stmt->val->ret_type)) {
    return nullptr;
  }

  // Atomic operators not supported by LLVM, we implement them using CAS
  if (stmt->op_type == AtomicOpType::mul) {
    return atomic_op_using_cas(
        llvm_val[stmt->dest], llvm_val[stmt->val], [&](auto v1, auto v2) { return builder->CreateMul(v1, v2); },
        stmt->val->ret_type);
  }
  // Atomic compare-and-swap: lowers to a single LLVM cmpxchg. The instruction returns a {value, success}
  // struct; we project field 0 (the loaded prior value), matching CUDA atomicCAS / SPIR-V OpAtomicCompareExchange.
  // The user recovers success with `(returned == expected)`.
  if (stmt->op_type == AtomicOpType::cas) {
    QD_ASSERT(stmt->expected != nullptr);
    auto cmpxchg = builder->CreateAtomicCmpXchg(llvm_val[stmt->dest], llvm_val[stmt->expected], llvm_val[stmt->val],
                                                llvm::MaybeAlign(0), llvm::AtomicOrdering::SequentiallyConsistent,
                                                llvm::AtomicOrdering::SequentiallyConsistent);
    return builder->CreateExtractValue(cmpxchg, 0);
  }
  // Atomic operators supported by LLVM
  std::unordered_map<AtomicOpType, llvm::AtomicRMWInst::BinOp> bin_op;
  bin_op[AtomicOpType::add] = llvm::AtomicRMWInst::BinOp::Add;
  if (is_signed(stmt->val->ret_type)) {
    bin_op[AtomicOpType::min] = llvm::AtomicRMWInst::BinOp::Min;
    bin_op[AtomicOpType::max] = llvm::AtomicRMWInst::BinOp::Max;
  } else {
    bin_op[AtomicOpType::min] = llvm::AtomicRMWInst::BinOp::UMin;
    bin_op[AtomicOpType::max] = llvm::AtomicRMWInst::BinOp::UMax;
  }
  bin_op[AtomicOpType::bit_and] = llvm::AtomicRMWInst::BinOp::And;
  bin_op[AtomicOpType::bit_or] = llvm::AtomicRMWInst::BinOp::Or;
  bin_op[AtomicOpType::bit_xor] = llvm::AtomicRMWInst::BinOp::Xor;
  bin_op[AtomicOpType::xchg] = llvm::AtomicRMWInst::BinOp::Xchg;
  QD_ASSERT(bin_op.find(stmt->op_type) != bin_op.end());
  return builder->CreateAtomicRMW(bin_op.at(stmt->op_type), llvm_val[stmt->dest], llvm_val[stmt->val],
                                  llvm::MaybeAlign(0), llvm::AtomicOrdering::SequentiallyConsistent,
                                  kernel_atomic_syncscope(llvm_context, current_arch()));
}

llvm::Value *TaskCodeGenLLVM::atomic_op_using_cas(llvm::Value *dest,
                                                  llvm::Value *val,
                                                  std::function<llvm::Value *(llvm::Value *, llvm::Value *)> op,
                                                  const DataType &type) {
  using namespace llvm;
  BasicBlock *body = BasicBlock::Create(*llvm_context, "while_loop_body", func);
  BasicBlock *after_loop = BasicBlock::Create(*llvm_context, "after_while", func);

  builder->CreateBr(body);
  builder->SetInsertPoint(body);

  llvm::Value *old_val;

  {
    int bits = data_type_bits(type);
    llvm::PointerType *typeIntPtr = llvm::PointerType::getUnqual(*llvm_context);
    llvm::IntegerType *typeIntTy = get_integer_type(bits);

    old_val = builder->CreateLoad(val->getType(), dest);
    auto new_val = op(old_val, val);
    dest = builder->CreateBitCast(dest, typeIntPtr);
    auto atomicCmpXchg = builder->CreateAtomicCmpXchg(
        dest, builder->CreateBitCast(old_val, typeIntTy), builder->CreateBitCast(new_val, typeIntTy),
        llvm::MaybeAlign(0), AtomicOrdering::SequentiallyConsistent, AtomicOrdering::SequentiallyConsistent,
        kernel_atomic_syncscope(llvm_context, current_arch()));
    // Check whether CAS was succussful
    auto ok = builder->CreateExtractValue(atomicCmpXchg, 1);
    builder->CreateCondBr(builder->CreateNot(ok), body, after_loop);
  }

  builder->SetInsertPoint(after_loop);

  return old_val;
}

llvm::Value *TaskCodeGenLLVM::real_type_atomic(AtomicOpStmt *stmt) {
  if (!is_real(stmt->val->ret_type)) {
    return nullptr;
  }

  PrimitiveTypeID prim_type = stmt->val->ret_type->cast<PrimitiveType>()->type;
  AtomicOpType op = stmt->op_type;
  if (prim_type == PrimitiveTypeID::f16) {
    switch (op) {
      case AtomicOpType::add:
        return atomic_op_using_cas(
            llvm_val[stmt->dest], llvm_val[stmt->val], [&](auto v1, auto v2) { return builder->CreateFAdd(v1, v2); },
            stmt->val->ret_type);
      case AtomicOpType::max:
        return atomic_op_using_cas(
            llvm_val[stmt->dest], llvm_val[stmt->val], [&](auto v1, auto v2) { return builder->CreateMaxNum(v1, v2); },
            stmt->val->ret_type);
      case AtomicOpType::min:
        return atomic_op_using_cas(
            llvm_val[stmt->dest], llvm_val[stmt->val], [&](auto v1, auto v2) { return builder->CreateMinNum(v1, v2); },
            stmt->val->ret_type);
      default:
        break;
    }
  }

  switch (op) {
    case AtomicOpType::add:
      return builder->CreateAtomicRMW(llvm::AtomicRMWInst::FAdd, llvm_val[stmt->dest], llvm_val[stmt->val],
                                      llvm::MaybeAlign(0), llvm::AtomicOrdering::SequentiallyConsistent,
                                      kernel_atomic_syncscope(llvm_context, current_arch()));
    case AtomicOpType::min:
      return builder->CreateAtomicRMW(llvm::AtomicRMWInst::FMin, llvm_val[stmt->dest], llvm_val[stmt->val],
                                      llvm::MaybeAlign(0), llvm::AtomicOrdering::SequentiallyConsistent,
                                      kernel_atomic_syncscope(llvm_context, current_arch()));
    case AtomicOpType::max:
      return builder->CreateAtomicRMW(llvm::AtomicRMWInst::FMax, llvm_val[stmt->dest], llvm_val[stmt->val],
                                      llvm::MaybeAlign(0), llvm::AtomicOrdering::SequentiallyConsistent,
                                      kernel_atomic_syncscope(llvm_context, current_arch()));
    case AtomicOpType::mul:
      return atomic_op_using_cas(
          llvm_val[stmt->dest], llvm_val[stmt->val], [&](auto v1, auto v2) { return builder->CreateFMul(v1, v2); },
          stmt->val->ret_type);
    case AtomicOpType::xchg:
      // LLVM AtomicRMW Xchg accepts FP types directly since LLVM 14, lowering to the natively-atomic swap instruction
      // (CUDA atomicExch / AMDGPU buffer_atomic_swap / x86 xchg). f16 falls through to the f16 CAS-emulation block
      // above.
      return builder->CreateAtomicRMW(llvm::AtomicRMWInst::Xchg, llvm_val[stmt->dest], llvm_val[stmt->val],
                                      llvm::MaybeAlign(0), llvm::AtomicOrdering::SequentiallyConsistent);
    default:
      break;
  }

  return nullptr;
}

void TaskCodeGenLLVM::visit(AtomicOpStmt *stmt) {
  bool is_local = stmt->dest->is<AllocaStmt>();
  if (is_local) {
    QD_ERROR("Local atomics should have been demoted.");
  }
  llvm::Value *old_value;
  if (llvm::Value *result = optimized_reduction(stmt)) {
    old_value = result;
  } else if (llvm::Value *result = quant_type_atomic(stmt)) {
    old_value = result;
  } else if (llvm::Value *result = real_type_atomic(stmt)) {
    old_value = result;
  } else if (llvm::Value *result = integral_type_atomic(stmt)) {
    old_value = result;
  } else {
    QD_NOT_IMPLEMENTED
  }
  llvm_val[stmt] = old_value;
}

void TaskCodeGenLLVM::visit(GlobalPtrStmt *stmt) {
  QD_ERROR("Global Ptrs should have been lowered.");
}

void TaskCodeGenLLVM::visit(GlobalStoreStmt *stmt) {
  QD_ASSERT(llvm_val[stmt->val]);
  QD_ASSERT(llvm_val[stmt->dest]);
  auto ptr_type = stmt->dest->ret_type->as<PointerType>();
  if (ptr_type->is_bit_pointer()) {
    auto pointee_type = ptr_type->get_pointee_type();
    auto snode = stmt->dest->as<GetChStmt>()->input_snode;
    if (snode->type == SNodeType::bit_struct) {
      QD_ERROR(
          "Bit struct stores with type {} should have been handled by "
          "BitStructStoreStmt.",
          pointee_type->to_string());
    }
    if (auto qit = pointee_type->cast<QuantIntType>()) {
      store_quant_int(llvm_val[stmt->dest], tlctx->get_data_type(snode->physical_type), qit, llvm_val[stmt->val], true);
    } else if (auto qfxt = pointee_type->cast<QuantFixedType>()) {
      store_quant_fixed(llvm_val[stmt->dest], tlctx->get_data_type(snode->physical_type), qfxt, llvm_val[stmt->val],
                        true);
    } else {
      QD_NOT_IMPLEMENTED;
    }
  } else {
    builder->CreateStore(llvm_val[stmt->val], llvm_val[stmt->dest]);
  }
}

llvm::Value *TaskCodeGenLLVM::create_intrinsic_load(llvm::Value *ptr, llvm::Type *ty) {
  QD_NOT_IMPLEMENTED;
}

void TaskCodeGenLLVM::create_global_load(GlobalLoadStmt *stmt, bool should_cache_as_read_only) {
  auto ptr = llvm_val[stmt->src];
  auto ptr_type = stmt->src->ret_type->as<PointerType>();
  // `is_volatile` and `should_cache_as_read_only` are mutually contradictory: the read-only path attaches
  // `!invariant.load` (or, on CUDA, lowers to `__ldg`), both of which give the optimiser license to hoist /
  // reuse the load.  Volatile loads are reserved for spin-wait patterns where exactly the opposite is required;
  // the frontend never plumbs both flags, so this is just a defensive guard against a future caller making the
  // mistake.
  QD_ASSERT(!(stmt->is_volatile && should_cache_as_read_only));
  if (ptr_type->is_bit_pointer()) {
    auto val_type = ptr_type->get_pointee_type();
    auto get_ch = stmt->src->as<GetChStmt>();
    auto physical_type = tlctx->get_data_type(get_ch->input_snode->physical_type);
    auto [byte_ptr, bit_offset] = load_bit_ptr(ptr);
    // Volatile loads on quant-bit-packed snodes are not meaningful (the pointed-to physical word is shared by
    // many quant fields, so a volatile read of the whole word does not give per-field volatile semantics).  No
    // public Python API exposes the combination today, so reject it eagerly rather than emit silently-wrong code.
    QD_ASSERT(!stmt->is_volatile);
    auto physical_value = should_cache_as_read_only ? create_intrinsic_load(byte_ptr, physical_type)
                                                    : builder->CreateLoad(physical_type, byte_ptr);
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
    if (should_cache_as_read_only) {
      llvm_val[stmt] = create_intrinsic_load(ptr, tlctx->get_data_type(stmt->ret_type));
    } else {
      auto *load = builder->CreateLoad(tlctx->get_data_type(stmt->ret_type), ptr);
      // LLVM's `setVolatile(true)` lowers to `ld.volatile.global` on PTX (for generic / addrspace(1) pointers)
      // and to `global_load_*` with the optimiser inhibited from hoisting / reusing the load on AMDGPU.  Both
      // backends treat this as the canonical "always re-read from memory" primitive.
      if (stmt->is_volatile) {
        load->setVolatile(true);
      }
      llvm_val[stmt] = load;
    }
  }
}

void TaskCodeGenLLVM::visit(GlobalLoadStmt *stmt) {
  create_global_load(stmt, false);
}

std::string TaskCodeGenLLVM::get_runtime_snode_name(SNode *snode) {
  if (snode->type == SNodeType::root) {
    return "Root";
  } else if (snode->type == SNodeType::dense) {
    return "Dense";
  } else if (snode->type == SNodeType::dynamic) {
    return "Dynamic";
  } else if (snode->type == SNodeType::pointer) {
    return "Pointer";
  } else if (snode->type == SNodeType::hash) {
    return "Hash";
  } else if (snode->type == SNodeType::bitmasked) {
    return "Bitmasked";
  } else if (snode->type == SNodeType::bit_struct) {
    return "BitStruct";
  } else if (snode->type == SNodeType::quant_array) {
    return "QuantArray";
  } else {
    QD_P(snode_type_name(snode->type));
    QD_NOT_IMPLEMENTED
  }
}

llvm::Value *TaskCodeGenLLVM::call(SNode *snode,
                                   llvm::Value *node_ptr,
                                   const std::string &method,
                                   const std::vector<llvm::Value *> &arguments) {
  auto prefix = get_runtime_snode_name(snode);
  auto s = emit_struct_meta(snode);
  auto s_ptr = builder->CreateBitCast(s, llvm::PointerType::getUnqual(*llvm_context));

  node_ptr = builder->CreateBitCast(node_ptr, llvm::PointerType::getUnqual(*llvm_context));

  std::vector<llvm::Value *> func_arguments{s_ptr, node_ptr};

  func_arguments.insert(func_arguments.end(), arguments.begin(), arguments.end());

  return call(prefix + "_" + method, std::move(func_arguments));
}

llvm::Function *TaskCodeGenLLVM::get_struct_function(const std::string &name, int tree_id) {
  used_tree_ids.insert(tree_id);
  auto f = tlctx->get_struct_function(name, tree_id);
  if (!f) {
    QD_ERROR("Struct function {} not found.", name);
  }
  f = llvm::cast<llvm::Function>(
      module->getOrInsertFunction(name, f->getFunctionType(), f->getAttributes()).getCallee());
  return f;
}

template <typename... Args>
llvm::Value *TaskCodeGenLLVM::call_struct_func(int tree_id, const std::string &func_name, Args &&...args) {
  auto func = get_struct_function(func_name, tree_id);
  auto arglist = std::vector<llvm::Value *>({args...});
  check_func_call_signature(func->getFunctionType(), func->getName(), arglist, builder.get());
  return builder->CreateCall(func, arglist);
}

void TaskCodeGenLLVM::visit(GetRootStmt *stmt) {
  if (stmt->root() == nullptr)
    llvm_val[stmt] = builder->CreateBitCast(
        get_root(SNodeTree::kFirstID),
        llvm::PointerType::get(
            StructCompilerLLVM::get_llvm_node_type(module.get(), prog->get_snode_root(SNodeTree::kFirstID)), 0));
  else
    llvm_val[stmt] = builder->CreateBitCast(
        get_root(stmt->root()->get_snode_tree_id()),
        llvm::PointerType::get(StructCompilerLLVM::get_llvm_node_type(module.get(), stmt->root()), 0));
}

void TaskCodeGenLLVM::visit(LinearizeStmt *stmt) {
  llvm::Value *val = tlctx->get_constant(0);
  for (int i = 0; i < (int)stmt->inputs.size(); i++) {
    val = builder->CreateAdd(builder->CreateMul(val, tlctx->get_constant(stmt->strides[i])), llvm_val[stmt->inputs[i]]);
  }
  llvm_val[stmt] = val;
}

void TaskCodeGenLLVM::visit(IntegerOffsetStmt *stmt){QD_NOT_IMPLEMENTED}

llvm::Value *TaskCodeGenLLVM::create_bit_ptr(llvm::Value *byte_ptr, llvm::Value *bit_offset) {
  // 1. define the bit pointer struct (X=8/16/32/64)
  // struct bit_pointer_X {
  //    iX* byte_ptr;
  //    i32 bit_offset;
  // };
  QD_ASSERT(bit_offset->getType()->isIntegerTy(32));
  auto struct_type = llvm::StructType::get(*llvm_context, {byte_ptr->getType(), bit_offset->getType()});
  // 2. allocate the bit pointer struct
  auto bit_ptr = create_entry_block_alloca(struct_type);
  // 3. store `byte_ptr`
  builder->CreateStore(byte_ptr,
                       builder->CreateGEP(struct_type, bit_ptr, {tlctx->get_constant(0), tlctx->get_constant(0)}));
  // 4. store `bit_offset
  builder->CreateStore(bit_offset,
                       builder->CreateGEP(struct_type, bit_ptr, {tlctx->get_constant(0), tlctx->get_constant(1)}));
  return bit_ptr;
}

std::tuple<llvm::Value *, llvm::Value *> TaskCodeGenLLVM::load_bit_ptr(llvm::Value *bit_ptr) {
  // FIXME: get ptr_ty from quadrants instead of llvm.
  llvm::Type *ptr_ty = nullptr;
  if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(bit_ptr))
    ptr_ty = AI->getAllocatedType();
  QD_ASSERT(ptr_ty);
  auto *struct_ty = llvm::cast<llvm::StructType>(ptr_ty);
  auto byte_ptr =
      builder->CreateLoad(struct_ty->getElementType(0),
                          builder->CreateGEP(ptr_ty, bit_ptr, {tlctx->get_constant(0), tlctx->get_constant(0)}));
  auto bit_offset =
      builder->CreateLoad(struct_ty->getElementType(1),
                          builder->CreateGEP(ptr_ty, bit_ptr, {tlctx->get_constant(0), tlctx->get_constant(1)}));

  return std::make_tuple(byte_ptr, bit_offset);
}

void TaskCodeGenLLVM::visit(SNodeLookupStmt *stmt) {
  llvm::Value *parent = nullptr;
  parent = llvm_val[stmt->input_snode];
  QD_ASSERT(parent);
  auto snode = stmt->snode;
  if (snode->type == SNodeType::root) {
    // FIXME: get parent_type from quadrants instead of llvm.
    llvm::Type *parent_ty = builder->getInt8Ty();
    if (auto bit_cast = llvm::dyn_cast<llvm::BitCastInst>(parent)) {
      parent_ty = bit_cast->getDestTy();
      if (auto ptr_ty = llvm::dyn_cast<llvm::PointerType>(parent_ty)) {
        QD_NOT_IMPLEMENTED;
      }
    }
    llvm_val[stmt] = builder->CreateGEP(parent_ty, parent, llvm_val[stmt->input_index]);
  } else if (snode->type == SNodeType::dense || snode->type == SNodeType::pointer ||
             snode->type == SNodeType::dynamic || snode->type == SNodeType::bitmasked) {
    if (stmt->activate) {
      call(snode, llvm_val[stmt->input_snode], "activate", {llvm_val[stmt->input_index]});
    }
    llvm_val[stmt] = call(snode, llvm_val[stmt->input_snode], "lookup_element", {llvm_val[stmt->input_index]});
  } else if (snode->type == SNodeType::bit_struct) {
    llvm_val[stmt] = parent;
  } else if (snode->type == SNodeType::quant_array) {
    auto element_num_bits = snode->dt->as<QuantArrayType>()->get_element_num_bits();
    auto offset = tlctx->get_constant(element_num_bits);
    offset = builder->CreateMul(offset, llvm_val[stmt->input_index]);
    llvm_val[stmt] = create_bit_ptr(llvm_val[stmt->input_snode], offset);
  } else {
    QD_INFO(snode_type_name(snode->type));
    QD_NOT_IMPLEMENTED
  }
}

void TaskCodeGenLLVM::visit(GetChStmt *stmt) {
  if (stmt->input_snode->type == SNodeType::quant_array) {
    llvm_val[stmt] = llvm_val[stmt->input_ptr];
  } else if (stmt->ret_type->as<PointerType>()->is_bit_pointer()) {
    auto bit_struct = stmt->input_snode->dt->cast<BitStructType>();
    auto bit_offset = bit_struct->get_member_bit_offset(stmt->output_snode->id_in_bit_struct);
    auto offset = tlctx->get_constant(bit_offset);
    llvm_val[stmt] = create_bit_ptr(llvm_val[stmt->input_ptr], offset);
  } else {
    auto ch = call_struct_func(
        stmt->output_snode->get_snode_tree_id(), stmt->output_snode->get_ch_from_parent_func_name(),
        builder->CreateBitCast(llvm_val[stmt->input_ptr], llvm::PointerType::getUnqual(*llvm_context)));
    llvm_val[stmt] = builder->CreateBitCast(
        ch, llvm::PointerType::get(StructCompilerLLVM::get_llvm_node_type(module.get(), stmt->output_snode), 0));
  }
}

void TaskCodeGenLLVM::visit(MatrixPtrStmt *stmt) {
  if (stmt->offset_used_as_index()) {
    auto type = tlctx->get_data_type(stmt->origin->ret_type.ptr_removed());

    auto casted_ptr = builder->CreateBitCast(llvm_val[stmt->origin], llvm::PointerType::get(type, 0));

    llvm_val[stmt] = builder->CreateGEP(type, casted_ptr, {tlctx->get_constant(0), llvm_val[stmt->offset]});
  } else {
    // Access PtrOffset via: base_ptr + offset
    auto origin_address = builder->CreatePtrToInt(llvm_val[stmt->origin], llvm::Type::getInt64Ty(*llvm_context));
    auto address_offset = builder->CreateSExt(llvm_val[stmt->offset], llvm::Type::getInt64Ty(*llvm_context));
    auto target_address = builder->CreateAdd(origin_address, address_offset);
    auto dt = stmt->ret_type.ptr_removed();
    llvm_val[stmt] = builder->CreateIntToPtr(target_address, llvm::PointerType::get(tlctx->get_data_type(dt), 0));
  }
}

void TaskCodeGenLLVM::visit(ExternalPtrStmt *stmt) {
  // Index into ndarray struct
  DataType operand_dtype = stmt->base_ptr->ret_type.ptr_removed()
                               ->as<StructType>()
                               ->get_element_type(std::array<int, 1>{1})
                               ->as<PointerType>()
                               ->get_pointee_type();
  auto arg_type = operand_dtype;
  if (operand_dtype->is<TensorType>()) {
    arg_type = operand_dtype->as<TensorType>()->get_element_type();
  }
  auto ptr_type = TypeFactory::get_instance().get_pointer_type(arg_type);
  auto members = stmt->base_ptr->ret_type.ptr_removed()->as<StructType>()->elements();
  bool needs_grad = members.size() > TypeFactory::GRAD_PTR_POS_IN_NDARRAY;
  auto struct_type =
      tlctx->get_data_type(TypeFactory::get_instance().get_ndarray_struct_type(arg_type, stmt->ndim, needs_grad));
  auto *gep = builder->CreateGEP(struct_type, llvm_val.at(stmt->base_ptr),
                                 {tlctx->get_constant(0), tlctx->get_constant(int(stmt->is_grad) + 1)});
  auto *ptr_val = builder->CreateLoad(tlctx->get_data_type(ptr_type), gep);

  int num_indices = stmt->indices.size();
  std::vector<llvm::Value *> sizes(num_indices);
  auto dt = stmt->ret_type.ptr_removed();
  int num_element_indices = dt->is<TensorType>() ? 0 : stmt->element_shape.size();

  /*
    ExternalPtrStmt can be divided into "outter" and "inner" parts.

    For example, "x" is an Ndarray with shape = (5, 5, 6), m=2, n=3.
    Indexing to a single element of "x" is of form: x[i, j, k][m, n]

    The "outter" part is x[i, j, k], and the "inner" part is [m, n].
    Shape of the inner part is known at compile time, stored in its ret_type.
    Shape of the outter part is determined at runtime, passed from the
    "extra_args".

    "num_indices - num_element_indices" gives how many "extra_args" to read from
  */
  int num_array_args = num_indices - num_element_indices;
  const size_t element_shape_index_offset = num_array_args;

  for (int i = 0; i < num_array_args; i++) {
    auto raw_arg = builder->CreateGEP(
        struct_type, llvm_val[stmt->base_ptr],
        {tlctx->get_constant(0), tlctx->get_constant(TypeFactory::SHAPE_POS_IN_NDARRAY), tlctx->get_constant(i)});
    raw_arg = builder->CreateLoad(tlctx->get_data_type(PrimitiveType::i32), raw_arg);
    sizes[i] = raw_arg;
  }

  auto linear_index = tlctx->get_constant(0);
  size_t size_var_index = 0;
  for (int i = 0; i < num_indices; i++) {
    if (i >= element_shape_index_offset && i < element_shape_index_offset + num_element_indices) {
      // Indexing TensorType-elements
      llvm::Value *size_var = tlctx->get_constant(stmt->element_shape[i - element_shape_index_offset]);
      linear_index = builder->CreateMul(linear_index, size_var);
    } else {
      // Indexing array dimensions
      linear_index = builder->CreateMul(linear_index, sizes[size_var_index++]);
    }
    linear_index = builder->CreateAdd(linear_index, llvm_val[stmt->indices[i]]);
  }
  QD_ASSERT(size_var_index == num_indices - num_element_indices);

  /*
    llvm::GEP implicitly indicates alignment when used upon llvm::VectorType.
    For example:

      "getelementptr <10 x i32>* %1, 0, 1" is interpreted as "%1 + 16(aligned)"

    However, this does not fit with Quadrants's Ndarray semantics. We will have
    to do pointer arithmetics to manually calculate the offset.
  */
  if (operand_dtype->is<TensorType>()) {
    // Access PtrOffset via: base_ptr + offset * sizeof(element)

    auto address_offset = builder->CreateSExt(linear_index, llvm::Type::getInt64Ty(*llvm_context));

    auto stmt_ret_type = stmt->ret_type.ptr_removed();
    if (stmt_ret_type->is<TensorType>()) {
      // This case corresponds to outter indexing only
      // The stride for linear_index is num_elements() in TensorType.
      address_offset = builder->CreateMul(
          address_offset,
          tlctx->get_constant(get_data_type<int64>(), stmt_ret_type->cast<TensorType>()->get_num_elements()));
    } else {
      // This case corresponds to outter + inner indexing
      // Since both outter and inner indices are linearized into linear_index,
      // the stride for linear_index is 1, and there's nothing to do here.
    }

    auto ret_ptr = builder->CreateGEP(tlctx->get_data_type(arg_type), ptr_val, address_offset);
    llvm_val[stmt] = builder->CreateBitCast(ret_ptr, llvm::PointerType::get(tlctx->get_data_type(dt), 0));

  } else {
    auto base_ty = tlctx->get_data_type(dt);
    auto base = builder->CreateBitCast(ptr_val, llvm::PointerType::get(base_ty, 0));

    llvm_val[stmt] = builder->CreateGEP(base_ty, base, linear_index);
  }
}

void TaskCodeGenLLVM::visit(ExternalTensorShapeAlongAxisStmt *stmt) {
  const auto arg_id = stmt->arg_id;
  const auto axis = stmt->axis;
  auto extended_arg_id = arg_id;
  extended_arg_id.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
  extended_arg_id.push_back(axis);
  llvm_val[stmt] = get_struct_arg(extended_arg_id, /*create_load=*/true);
}

void TaskCodeGenLLVM::visit(ExternalTensorBasePtrStmt *stmt) {
  auto arg_id = stmt->arg_id;
  int pos = stmt->is_grad ? TypeFactory::GRAD_PTR_POS_IN_NDARRAY : TypeFactory::DATA_PTR_POS_IN_NDARRAY;
  arg_id.push_back(pos);
  llvm_val[stmt] = get_struct_arg(arg_id, /*create_load=*/true);
}

std::string TaskCodeGenLLVM::init_offloaded_task_function(OffloadedStmt *stmt, std::string suffix) {
  current_loop_reentry = nullptr;
  current_while_after_loop = nullptr;

  // Reset per-task heap-adstack state. `ad_stack_per_thread_stride_*` and `ad_stack_offsets_` are (re)populated by the
  // pre-scan below; `ad_stack_heap_base_*_llvm_` is emitted lazily when the first AdStack* stmt of this task fires.
  // Clearing is important because a kernel with multiple offloaded tasks shares this visitor instance and a stale
  // map/base from the previous task would either grow stride unboundedly or (worse) reuse an SSA value from a different
  // function, tripping `verifyFunction` inside `finalize_offloaded_task_function`.
  ad_stack_per_thread_stride_ = 0;
  ad_stack_per_thread_stride_float_ = 0;
  ad_stack_per_thread_stride_int_ = 0;
  ad_stack_offsets_.clear();
  ad_stack_allocas_info_.clear();
  ad_stack_size_exprs_.clear();
  ad_stack_heap_base_float_llvm_ = nullptr;
  ad_stack_heap_base_int_llvm_ = nullptr;
  ad_stack_stride_llvm_ = nullptr;
  ad_stack_stride_float_llvm_ = nullptr;
  ad_stack_stride_int_llvm_ = nullptr;
  ad_stack_offsets_ptr_llvm_ = nullptr;
  ad_stack_max_sizes_ptr_llvm_ = nullptr;
  ad_stack_count_alloca_llvm_.clear();
  ad_stack_row_id_var_float_llvm_ = nullptr;
  ad_stack_bootstrap_pushes_.clear();
  ad_stack_lazy_float_allocas_.clear();
  ad_stack_static_bound_expr_.reset();

  // Run the shared static-adstack analysis. Returns the LCA of every f32 push/load-top site, the autodiff-bootstrap
  // const-init push set, and an optional captured `StaticBoundExpr` when a single recognized gate sits on the
  // LCA-to-root chain. The SNode descriptor resolver walks the leaf SNode's parent chain to identify the owning tree,
  // then reads the LLVM declaration-order offsets the runtime struct compiler already populated on the live SNode tree
  // (`SNode::offset_bytes_in_parent_cell` set by `StructCompilerLLVM::generate_types`, mirrored by the host-side reader
  // `LlvmProgramImpl::get_field_in_tree_offset`). Reading those fields directly keeps the captured base offset / cell
  // stride byte-correct against the LLVM runtime layout, including the multi-leaf dense case where `qd.root.dense(qd.i,
  // n).place(field_f64, field_f32)` has children of mixed sizes. The SPIR-V struct compiler `compile_snode_structs`
  // sorts dense children by ascending size and would land on the wrong offset here, plus it mutates
  // `offset_bytes_in_parent_cell` and `cell_size_bytes` on the shared SNode tree as a side effect (corrupting later
  // readers in `dlpack_funcs.cpp` and `field_info.cpp`). Trees outside the kernel's `program->snode_trees_` range or
  // non-dense parents fall through to nullopt and the analysis rejects the gate (worst-case sizing in the runtime
  // caller).
  auto snode_resolver = [&](const SNode *leaf, const SNode *dense) -> std::optional<SNodeFieldDescriptor> {
    if (leaf == nullptr || dense == nullptr || prog == nullptr) {
      return std::nullopt;
    }
    const SNode *root_snode = dense->parent;
    if (root_snode == nullptr) {
      return std::nullopt;
    }
    // Find which `snode_tree_id` this root belongs to. `program->get_snode_root(id)` returns the SNode for tree `id`;
    // iterate until we find a match. Tree counts are small (single digits in every observed kernel) so the linear scan
    // is cheap and avoids needing a public reverse-lookup API on `Program`. Bound the scan with
    // `prog->get_snode_tree_size()` - `Program::get_snode_root` is a raw `snode_trees_[tree_id]->root()` with no bounds
    // check, so an unbounded loop would be `std::vector::operator[]` OOB undefined behaviour on programs whose tree-id
    // space is smaller than the captured chain expects (stale SNode references, recycled tree slots, offline-cache
    // restore mismatches). The SPIR-V analog uses a bounded `snode_to_root_` map; mirror that safety here. Continue
    // (rather than break) past nullptr slots to handle recycled-tree-id holes from `free_snode_tree_ids_`.
    int matched_tree_id = -1;
    for (int id = SNodeTree::kFirstID; id < prog->get_snode_tree_size(); ++id) {
      SNode *root_for_id = prog->get_snode_root(id);
      if (root_for_id == nullptr) {
        continue;
      }
      if (root_for_id == root_snode) {
        matched_tree_id = id;
        break;
      }
    }
    if (matched_tree_id < 0) {
      return std::nullopt;
    }
    SNodeFieldDescriptor desc;
    desc.root_id = matched_tree_id;
    // Combined byte offset: dense's offset within its single root cell plus the leaf's offset within the dense's
    // per-cell layout. Both fields are populated by `StructCompilerLLVM::generate_types` before any kernel codegen
    // runs, in declaration order matching the LLVM accessors the main kernel emits.
    desc.byte_base_offset =
        static_cast<uint32_t>(dense->offset_bytes_in_parent_cell + leaf->offset_bytes_in_parent_cell);
    // Per-cell stride for the dense parent. `cell_size_bytes` is the size of one element of the dense's child struct
    // (set on the dense by `StructCompilerLLVM::generate_types`).
    desc.byte_cell_stride = static_cast<uint32_t>(dense->cell_size_bytes);
    // Iteration count: product of `num_elements_from_root` over the dense's extractors. Mirrors the SPIR-V compiler's
    // `total_num_cells_from_root` formula in `snode_struct_compiler.cpp` but reads the extractor metadata from the live
    // SNode tree (`SNode::extractors[i].num_elements_from_root`, populated by `StructCompiler::infer_snode_properties`)
    // instead of going through the SPIR-V descriptor cache.
    uint64_t iter_count = 1;
    for (const auto &e : dense->extractors) {
      iter_count *= static_cast<uint64_t>(e.num_elements_from_root);
    }
    desc.iter_count = static_cast<uint32_t>(iter_count);
    return desc;
  };
  // CPU LLVM goes through `make_cpu_multithreaded_range_for` in `offload_to_executable`, which rewrites the user
  // loop's `[begin_value, end_value)` into per-thread chunks before codegen runs. The atomic row counter the
  // codegen emits is shared across every chunk of the same task, so the total claim count is the original
  // pre-chunk loop trip count, not the per-chunk subrange. Signal that to the analyzer so it skips filling
  // `bound_expr.loop_iter_static` on CPU and the runtime falls back to the unclipped reducer count there. CUDA
  // and AMDGPU dispatch one thread per iteration without chunking, so their per-task `[begin_value, end_value)`
  // matches the user loop and the analyzer can fill the field.
  const bool task_range_is_original_loop = !arch_is_cpu(compile_config.arch);
  auto adstack_analysis = analyze_adstack_static_bounds(
      stmt, snode_resolver, compile_config.ad_stack_sparse_threshold_bytes, task_range_is_original_loop);
  ad_stack_bootstrap_pushes_ = std::move(adstack_analysis.bootstrap_pushes);
  ad_stack_lca_block_float_ir_ = adstack_analysis.lca_block_float;
  ad_stack_static_bound_expr_ = adstack_analysis.bound_expr;

  // Pre-scan the task body for every `AdStackAllocaStmt` before any codegen runs. Each alloca claims a fixed slot
  // inside its kind's per-thread slice (`HeapKind::Float` slot in the float heap, `HeapKind::Int` slot in the int
  // heap); the kind classification is recorded into `info.heap_kind` and `visit(AdStackAllocaStmt)` routes the base
  // computation per kind via `ad_stack_heap_base_float_llvm_` / `ad_stack_heap_base_int_llvm_` and the matching
  // strides. The shared analysis output (LCA, bootstrap pushes, captured `bound_expr`) propagates to
  // `current_task->ad_stack` so the host launcher can dispatch the per-arch reducer. Sizes are rounded up to 8 bytes
  // so `stack_top_primal`'s `stack + sizeof(u64) + idx * 2 * element_size` math stays naturally aligned for every
  // element type the IR may emit (i8 / u1 pack especially, on which the raw `size_in_bytes()` is otherwise unaligned).
  {
    auto align_up_8 = [](std::size_t n) -> std::size_t { return (n + 7u) & ~std::size_t{7u}; };
    std::function<void(IRNode *)> scan = [&](IRNode *node) {
      if (auto *blk = dynamic_cast<Block *>(node)) {
        for (auto &s : blk->statements)
          scan(s.get());
      } else if (auto *alloca = dynamic_cast<AdStackAllocaStmt *>(node)) {
        alloca->stack_id = static_cast<int>(ad_stack_offsets_.size());
        ad_stack_offsets_.push_back(ad_stack_per_thread_stride_);
        ad_stack_per_thread_stride_ += align_up_8(alloca->size_in_bytes());
        const bool is_float = alloca->ret_type == PrimitiveType::f32 || alloca->ret_type == PrimitiveType::f64;
        if (is_float) {
          ad_stack_per_thread_stride_float_ += align_up_8(alloca->size_in_bytes());
        } else {
          ad_stack_per_thread_stride_int_ += align_up_8(alloca->size_in_bytes());
        }
        // Mirror the compile-time sizing into the per-task metadata: the launcher uses `allocas[stack_id]` to publish
        // stride / offset / max_size values into the per-launch runtime buffers regardless of whether the symbolic
        // `size_expr` survived the offline-cache round-trip. When a cached kernel is loaded with its `size_exprs`
        // dropped (the SerializedSizeExpr blob is keyed off the IR shape and is not part of the cache schema), the
        // device-side sizer falls back to `max_size_compile_time` published here as the conservative ceiling.
        AdStackAllocaInfo info;
        info.offset = ad_stack_offsets_.back();
        info.max_size_compile_time = alloca->max_size;
        info.entry_size_bytes = alloca->entry_size_in_bytes();
        info.heap_kind = is_float ? AdStackAllocaInfo::HeapKind::Float : AdStackAllocaInfo::HeapKind::Int;
        ad_stack_allocas_info_.push_back(info);
        ad_stack_size_exprs_.push_back(alloca->size_expr ? alloca->size_expr->serialize() : SerializedSizeExpr{});
      } else if (auto *if_stmt = dynamic_cast<IfStmt *>(node)) {
        if (if_stmt->true_statements)
          scan(if_stmt->true_statements.get());
        if (if_stmt->false_statements)
          scan(if_stmt->false_statements.get());
      } else if (auto *range_for = dynamic_cast<RangeForStmt *>(node)) {
        scan(range_for->body.get());
      } else if (auto *struct_for = dynamic_cast<StructForStmt *>(node)) {
        // Defensive: struct_for offloads encode the loop in the OffloadedStmt's `task_type` rather than as a nested
        // `StructForStmt` in the body, so walking the offload body never lands on a `StructForStmt` from production
        // Python kernels today. Recurse anyway to keep this pre-scan symmetric with `analyze_adstack_static_bounds`'s
        // `walk_ir` helper - if a future IR refactor introduces a `StructForStmt` between the offload root and an
        // `AdStackAllocaStmt`, the alloca's `stack_id` would otherwise stay unassigned and the codegen-emitted base
        // computation would index `ad_stack_offsets_` out of bounds.
        scan(struct_for->body.get());
      } else if (auto *mesh_for = dynamic_cast<MeshForStmt *>(node)) {
        // Same rationale as the `StructForStmt` branch above: mesh_for offloads encode the loop in `task_type`. Recurse
        // for symmetry with `analyze_adstack_static_bounds::walk_ir`.
        scan(mesh_for->body.get());
      } else if (auto *while_stmt = dynamic_cast<WhileStmt *>(node)) {
        scan(while_stmt->body.get());
      }
    };
    if (stmt->body) {
      scan(stmt->body.get());
    }
  }

  task_function_type =
      llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_context), {llvm::PointerType::get(context_ty, 0)}, false);

  auto task_kernel_name =
      stmt->loop_name.empty()
          ? fmt::format("{}_{}_{}{}", kernel_name, task_codegen_id, stmt->task_name(), suffix)
          : fmt::format("{}_{}_{}_{}{}", kernel_name, task_codegen_id, stmt->loop_name, stmt->task_name(), suffix);
  func = llvm::Function::Create(task_function_type, llvm::Function::ExternalLinkage, task_kernel_name, module.get());

  current_task = std::make_unique<OffloadedTask>(task_kernel_name);
  // Carry the per-task graph_do_while level tag for all LLVM backends (CPU/CUDA/AMDGPU). Set here, in the shared
  // task-init path, rather than per-backend (unlike stream_parallel_group_id which is GPU-only), because the
  // graph_do_while host fallback runs on CPU too.
  current_task->graph_do_while_level_id = stmt->graph_do_while_level_id;
  // Pre-register the per-task AdStackSizingInfo so the registry id is assigned BEFORE codegen visits any
  // `AdStackPushStmt`, letting it bake the immediate. Metadata (`allocated_max_sizes` + `size_exprs`) is filled in at
  // `finalize_offloaded_task_function` time after the alloca scan completes; the registry call is idempotent on the
  // same identity_key (raw `&current_task->ad_stack` address, never derefed) so the second call updates the entry in
  // place. `kernel_name` and `task_id_in_kernel` are also stashed on the ad_stack so the offline-cache reload path
  // (`AdStackCache::ensure_runtime_registry_ids_for_max_reducer`) can re-derive the identity hash without parsing the
  // task function name. Skipping registration when `prog == nullptr` (C++-only tests) leaves `registry_id == 0`, which
  // the codegen-emitted cmpxchg short-circuits.
  if (prog != nullptr) {
    current_task->ad_stack.kernel_name = kernel_name;
    current_task->ad_stack.task_id_in_kernel = task_codegen_id;
    uint32_t id = prog->adstack_cache().register_adstack_sizing_info(
        static_cast<const void *>(&current_task->ad_stack), kernel_name, task_codegen_id, /*allocated_max_sizes=*/{},
        /*size_exprs=*/{});
    current_task->ad_stack.registry_id = id;
  }

  for (auto &arg : func->args()) {
    kernel_args.push_back(&arg);
  }
  kernel_args[0]->setName("context");
  if (kernel_argument_by_val())
    func->addParamAttr(0, llvm::Attribute::getWithByValType(*llvm_context, context_ty));
  // entry_block has all the allocas
  this->entry_block = llvm::BasicBlock::Create(*llvm_context, "entry", func);
  this->final_block = llvm::BasicBlock::Create(*llvm_context, "final", func);

  // The real function body
  func_body_bb = llvm::BasicBlock::Create(*llvm_context, "body", func);
  builder->SetInsertPoint(func_body_bb);
  return task_kernel_name;
}

void TaskCodeGenLLVM::finalize_offloaded_task_function() {
  if (!returned) {
    builder->CreateBr(final_block);
  } else {
    returned = false;
  }
  builder->SetInsertPoint(final_block);
  builder->CreateRetVoid();

  // Propagate per-thread adstack stride into the OffloadedTask so the host-side kernel launcher can
  // size `LlvmRuntimeExecutor::adstack_heap_` before dispatch. `static_num_threads` and the
  // dynamic-offset fields are populated by each backend's codegen after `grid_dim` / `block_dim`
  // are finalized (see codegen_cpu / codegen_cuda / codegen_amdgpu).
  if (current_task) {
    current_task->ad_stack.per_thread_stride = ad_stack_per_thread_stride_;
    current_task->ad_stack.per_thread_stride_float = ad_stack_per_thread_stride_float_;
    current_task->ad_stack.per_thread_stride_int = ad_stack_per_thread_stride_int_;
    current_task->ad_stack.allocas = ad_stack_allocas_info_;
    current_task->ad_stack.size_exprs = ad_stack_size_exprs_;
    current_task->ad_stack.bound_expr = ad_stack_static_bound_expr_;
    // Recognize `MaxOverRange` nodes the runtime can reduce in parallel via the dedicated max-reducer dispatch instead
    // of letting the per-thread sizer enumerate. Indexing matches `ad_stack_size_exprs_` (same iteration order as the
    // pre-scan above). Skip on CPU: the host evaluator's `MaxOverRange` loop in `program/adstack/eval.cpp` does the
    // same serial walk, and dispatching the runtime helper would only add per-launch setup cost (params blob encode,
    // body bytecode encode, observation bookkeeping, JIT call) with no compute parallelism to amortize. The host
    // evaluator handles every iteration count up to its own cap (`UINT32_MAX` on CPU; see `eval.cpp`). On CUDA /
    // AMDGPU the parallel reducer is the whole point of the dispatch and the recognizer stays active.
    if (!arch_is_cpu(compile_config.arch)) {
      current_task->ad_stack.max_reducer_specs = recognize_adstack_max_reducer_specs(ad_stack_size_exprs_);
    }
    // Snodes the task body mutates. Persisted on `OffloadedTask::snode_writes` so the LLVM
    // launcher can invalidate the per-task adstack metadata cache when a kernel that runs in
    // between mutated a SNode an enclosing `size_expr::FieldLoad` reads. Mirrors the SPIR-V
    // analogue in `spirv_codegen.cpp`. Sorted + deduplicated for stable serialisation.
    if (current_offload != nullptr) {
      auto snode_rw = irpass::analysis::gather_snode_read_writes(current_offload);
      current_task->snode_writes.reserve(snode_rw.second.size());
      for (auto *s : snode_rw.second) {
        if (s != nullptr) {
          current_task->snode_writes.push_back(s->id);
        }
      }
      std::sort(current_task->snode_writes.begin(), current_task->snode_writes.end());
      current_task->snode_writes.erase(
          std::unique(current_task->snode_writes.begin(), current_task->snode_writes.end()),
          current_task->snode_writes.end());
      // Ndarray args this task writes to. Same role as `snode_writes` but for ndarray data;
      // covers `size_expr::ExternalTensorRead` invalidation. The first element of each
      // `arg_id_path` key is the kernel-arg slot, which is what `Program::ndarray_data_gen_`
      // is keyed by (via the bound DeviceAllocation).
      auto arr_access = irpass::detect_external_ptr_access_in_task(current_offload);
      for (const auto &kv : arr_access) {
        if (kv.first.empty()) {
          continue;
        }
        const uint32_t access_bits = static_cast<uint32_t>(kv.second);
        if ((access_bits & static_cast<uint32_t>(irpass::ExternalPtrAccess::WRITE)) != 0) {
          current_task->arr_writes.push_back(kv.first.front());
        }
        if ((access_bits & static_cast<uint32_t>(irpass::ExternalPtrAccess::READ)) != 0) {
          current_task->arr_reads.push_back(kv.first.front());
        }
      }
      std::sort(current_task->arr_writes.begin(), current_task->arr_writes.end());
      current_task->arr_writes.erase(std::unique(current_task->arr_writes.begin(), current_task->arr_writes.end()),
                                     current_task->arr_writes.end());
      std::sort(current_task->arr_reads.begin(), current_task->arr_reads.end());
      current_task->arr_reads.erase(std::unique(current_task->arr_reads.begin(), current_task->arr_reads.end()),
                                    current_task->arr_reads.end());
    }
    // Register the per-task AdStackSizingInfo with the Program-side identity registry. The id is baked
    // into the lazy-claim overflow path's `cmpxchg(0, id)` so the host raise site can name the offending
    // kernel + task in its diagnostic message. Empty alloca list = no adstack pushes in this task; skip
    // registration to keep the registry compact.
    if (!current_task->ad_stack.allocas.empty() && prog != nullptr) {
      std::vector<int> allocated_max_sizes;
      allocated_max_sizes.reserve(current_task->ad_stack.allocas.size());
      for (const auto &a : current_task->ad_stack.allocas) {
        allocated_max_sizes.push_back(static_cast<int>(a.max_size_compile_time));
      }
      // Update the entry with the live metadata + per-alloca size_exprs. The size_exprs are copied into the registry so
      // the diagnose path can walk them without dereferencing the launcher's unstable `OffloadedTask::ad_stack` pointer
      // (freed by `current_task = nullptr` after by-value `offloaded_tasks.push_back(*current_task)`). Mirror the
      // identity-pair fields here too in case the task START registration above was skipped (no allocas at the time,
      // prog null, etc.).
      current_task->ad_stack.kernel_name = kernel_name;
      current_task->ad_stack.task_id_in_kernel = task_codegen_id;
      uint32_t id = prog->adstack_cache().register_adstack_sizing_info(
          static_cast<const void *>(&current_task->ad_stack), kernel_name, task_codegen_id,
          std::move(allocated_max_sizes), current_task->ad_stack.size_exprs);
      current_task->ad_stack.registry_id = id;
    }
  }

  // entry_block should jump to the body after all allocas are inserted
  builder->SetInsertPoint(entry_block);
  builder->CreateBr(func_body_bb);

  if (compile_config.print_kernel_llvm_ir) {
    static FileSequenceWriter writer("quadrants_kernel_generic_llvm_ir_{:04d}.ll", "unoptimized LLVM IR (generic)");
    writer.write(module.get());
  }
  QD_ASSERT(!llvm::verifyFunction(*func, &llvm::errs()));
  // QD_INFO("Kernel function verified.");
}

std::tuple<llvm::Value *, llvm::Value *> TaskCodeGenLLVM::get_range_for_bounds(OffloadedStmt *stmt) {
  llvm::Value *begin, *end;
  if (stmt->const_begin) {
    begin = tlctx->get_constant(stmt->begin_value);
  } else {
    auto begin_stmt = Stmt::make<GlobalTemporaryStmt>(stmt->begin_offset, PrimitiveType::i32);
    begin_stmt->accept(this);
    begin = builder->CreateLoad(tlctx->get_data_type(PrimitiveType::i32), llvm_val[begin_stmt.get()]);
  }
  if (stmt->const_end) {
    end = tlctx->get_constant(stmt->end_value);
  } else {
    auto end_stmt = Stmt::make<GlobalTemporaryStmt>(stmt->end_offset, PrimitiveType::i32);
    end_stmt->accept(this);
    end = builder->CreateLoad(tlctx->get_data_type(PrimitiveType::i32), llvm_val[end_stmt.get()]);
  }
  return std::tuple(begin, end);
}

void TaskCodeGenLLVM::create_offload_struct_for(OffloadedStmt *stmt) {
  using namespace llvm;
  // TODO: instead of constructing tons of LLVM IR, writing the logic in
  // runtime.cpp may be a cleaner solution. See
  // TaskCodeGenCPU::create_offload_range_for as an example.

  llvm::Function *body = nullptr;
  auto leaf_block = stmt->snode;

  // For a bit-vectorized loop over a quant array, we generate struct for on its
  // parent node (must be "dense") instead of itself for higher performance.
  if (stmt->is_bit_vectorized) {
    if (leaf_block->type == SNodeType::quant_array && leaf_block->parent->type == SNodeType::dense) {
      leaf_block = leaf_block->parent;
    } else {
      QD_ERROR(
          "A bit-vectorized struct-for must loop over a quant array with a "
          "dense parent");
    }
  }

  {
    // Create the loop body function
    auto guard = get_function_creation_guard({
        llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
        get_tls_buffer_type(),
        llvm::PointerType::get(get_runtime_type("Element"), 0),
        tlctx->get_data_type<int>(),
        tlctx->get_data_type<int>(),
    });

    body = guard.body;

    /* Function structure:
     *
     * function_body (entry):
     *   loop_index = lower_bound;
     *   tls_prologue()
     *   bls_prologue()
     *   goto loop_test
     *
     * loop_test:
     *   if (loop_index < upper_bound)
     *     goto loop_body
     *   else
     *     goto func_exit
     *
     * loop_body:
     *   initialize_coordinates()
     *   if (bitmasked voxel is active)
     *     goto struct_for_body
     *   else
     *     goto loop_body_tail
     *
     * struct_for_body:
     *   ... (Run codegen on the StructForStmt::body Quadrants Block)
     *   goto loop_body_tail
     *
     * loop_body_tail:
     *   loop_index += block_dim
     *   goto loop_test
     *
     * func_exit:
     *   bls_epilogue()
     *   tls_epilogue()
     *   return
     */
    auto loop_index_ty = llvm::Type::getInt32Ty(*llvm_context);
    auto loop_index = create_entry_block_alloca(loop_index_ty);

    RuntimeObject element("Element", this, builder.get(), get_arg(2));

    // Loop ranges
    auto lower_bound = get_arg(3);
    auto upper_bound = get_arg(4);

    parent_coordinates = element.get_ptr("pcoord");
    block_corner_coordinates = create_entry_block_alloca(physical_coordinate_ty);

    auto refine = get_struct_function(leaf_block->refine_coordinates_func_name(), leaf_block->get_snode_tree_id());
    // A block corner is the global coordinate/index of the lower-left corner
    // cell within that block, and is the same for all the cells within that
    // block.
    call(refine, parent_coordinates, block_corner_coordinates, tlctx->get_constant(0));

    if (stmt->tls_prologue) {
      stmt->tls_prologue->accept(this);
    }

    if (stmt->bls_prologue) {
      call("block_barrier");  // "__syncthreads()"
      stmt->bls_prologue->accept(this);
      call("block_barrier");  // "__syncthreads()"
    }

    auto [thread_idx, block_dim] = this->get_spmd_info();
    builder->CreateStore(builder->CreateAdd(thread_idx, lower_bound), loop_index);

    auto loop_test_bb = BasicBlock::Create(*llvm_context, "loop_test", func);
    auto loop_body_bb = BasicBlock::Create(*llvm_context, "loop_body", func);
    auto body_tail_bb = BasicBlock::Create(*llvm_context, "loop_body_tail", func);
    auto func_exit = BasicBlock::Create(*llvm_context, "func_exit", func);
    auto struct_for_body_bb = BasicBlock::Create(*llvm_context, "struct_for_body_body", func);

    auto lrg = make_loop_reentry_guard(this);
    current_loop_reentry = body_tail_bb;

    builder->CreateBr(loop_test_bb);

    {
      // loop_test:
      //   if (loop_index < upper_bound)
      //     goto loop_body;
      //   else
      //     goto func_exit

      builder->SetInsertPoint(loop_test_bb);
      auto cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT,
                                      builder->CreateLoad(loop_index_ty, loop_index), upper_bound);
      builder->CreateCondBr(cond, loop_body_bb, func_exit);
    }

    // ***********************
    // Begin loop_body_bb:
    builder->SetInsertPoint(loop_body_bb);

    // initialize the coordinates
    auto new_coordinates = create_entry_block_alloca(physical_coordinate_ty);

    call(refine, parent_coordinates, new_coordinates, builder->CreateLoad(loop_index_ty, loop_index));

    // For a bit-vectorized loop over a quant array, one more refine step is
    // needed to make final coordinates non-consecutive, since each thread will
    // process multiple coordinates via vectorization
    if (stmt->is_bit_vectorized) {
      refine = get_struct_function(stmt->snode->refine_coordinates_func_name(), stmt->snode->get_snode_tree_id());
      call(refine, new_coordinates, new_coordinates, tlctx->get_constant(0));
    }

    current_coordinates = new_coordinates;

    // exec_cond: safe-guard the execution of loop body:
    //  - if non-POT field dim exists, make sure we don't go out of bounds
    //  - if leaf block is bitmasked, make sure we only loop over active
    //    voxels
    auto exec_cond = tlctx->get_constant(true);
    auto coord_object = RuntimeObject(kLLVMPhysicalCoordinatesName, this, builder.get(), new_coordinates);

    if (leaf_block->type == SNodeType::bitmasked || leaf_block->type == SNodeType::pointer) {
      // test whether the current voxel is active or not
      auto is_active =
          call(leaf_block, element.get("element"), "is_active", {builder->CreateLoad(loop_index_ty, loop_index)});
      is_active = builder->CreateIsNotNull(is_active);
      exec_cond = builder->CreateAnd(exec_cond, is_active);
    }

    builder->CreateCondBr(exec_cond, struct_for_body_bb, body_tail_bb);

    {
      builder->SetInsertPoint(struct_for_body_bb);

      // The real loop body of the StructForStmt
      stmt->body->accept(this);

      builder->CreateBr(body_tail_bb);
    }

    {
      // body tail: increment loop_index and jump to loop_test
      builder->SetInsertPoint(body_tail_bb);

      create_increment(loop_index, block_dim);
      builder->CreateBr(loop_test_bb);

      builder->SetInsertPoint(func_exit);
    }

    if (stmt->bls_epilogue) {
      call("block_barrier");  // "__syncthreads()"
      stmt->bls_epilogue->accept(this);
      call("block_barrier");  // "__syncthreads()"
    }

    if (stmt->tls_epilogue) {
      stmt->tls_epilogue->accept(this);
    }
  }

  int list_element_size = std::min(leaf_block->max_num_elements(), (int64)quadrants_listgen_max_element_size);
  int num_splits = std::max(1, list_element_size / stmt->block_dim + (list_element_size % stmt->block_dim != 0));

  auto struct_for_func = get_runtime_function("parallel_struct_for");

  if (arch_is_gpu(current_arch())) {
    struct_for_func = llvm::cast<llvm::Function>(
        module
            ->getOrInsertFunction(tlctx->get_struct_for_func_name(stmt->tls_size), struct_for_func->getFunctionType(),
                                  struct_for_func->getAttributes())
            .getCallee());
    struct_for_tls_sizes.insert(stmt->tls_size);
  }
  // Loop over nodes in the element list, in parallel
  call(struct_for_func, get_context(), tlctx->get_constant(leaf_block->id), tlctx->get_constant(list_element_size),
       tlctx->get_constant(num_splits), body, tlctx->get_constant(stmt->tls_size),
       tlctx->get_constant(stmt->num_cpu_threads));
  // TODO: why do we need num_cpu_threads on GPUs?

  current_coordinates = nullptr;
  parent_coordinates = nullptr;
  block_corner_coordinates = nullptr;
}

void TaskCodeGenLLVM::visit(LoopIndexStmt *stmt) {
  if (stmt->loop->is<OffloadedStmt>() &&
      stmt->loop->as<OffloadedStmt>()->task_type == OffloadedStmt::TaskType::struct_for) {
    llvm::Type *struct_ty = nullptr;
    // FIXME: get struct_ty from quadrants instead of llvm.
    if (auto *alloca = llvm::dyn_cast<llvm::AllocaInst>(current_coordinates)) {
      struct_ty = alloca->getAllocatedType();
    }
    QD_ASSERT(struct_ty);
    auto *GEP = builder->CreateGEP(struct_ty, current_coordinates,
                                   {tlctx->get_constant(0), tlctx->get_constant(0), tlctx->get_constant(stmt->index)});
    if (stmt->index == 0 && !llvm::isa<llvm::GEPOperator>(GEP))
      GEP = builder->CreateBitCast(GEP, struct_ty->getPointerTo());
    llvm_val[stmt] = builder->CreateLoad(llvm::Type::getInt32Ty(*llvm_context), GEP);
  } else {
    llvm_val[stmt] =
        builder->CreateLoad(llvm::Type::getInt32Ty(*llvm_context), loop_vars_llvm[stmt->loop][stmt->index]);
  }
}

void TaskCodeGenLLVM::visit(LoopLinearIndexStmt *stmt) {
  if (stmt->loop->is<OffloadedStmt>() &&
      (stmt->loop->as<OffloadedStmt>()->task_type == OffloadedStmt::TaskType::struct_for ||
       stmt->loop->as<OffloadedStmt>()->task_type == OffloadedStmt::TaskType::mesh_for)) {
    llvm_val[stmt] = call("thread_idx");
  } else {
    QD_NOT_IMPLEMENTED;
  }
}

void TaskCodeGenLLVM::visit(BlockCornerIndexStmt *stmt) {
  if (stmt->loop->is<OffloadedStmt>() &&
      stmt->loop->as<OffloadedStmt>()->task_type == OffloadedStmt::TaskType::struct_for) {
    QD_ASSERT(block_corner_coordinates);
    // Make sure physical_coordinate_ty matches
    // struct PhysicalCoordinates {
    //   i32 val[quadrants_max_num_indices];
    // };
    QD_ASSERT(physical_coordinate_ty->isStructTy());
    auto physical_coordinate_ty_as_struct = llvm::cast<llvm::StructType>(physical_coordinate_ty);
    QD_ASSERT(physical_coordinate_ty_as_struct);
    QD_ASSERT(physical_coordinate_ty_as_struct->getNumElements() == 1);
    auto val_ty = physical_coordinate_ty_as_struct->getElementType(0);
    QD_ASSERT(val_ty->isArrayTy());
    auto val_ty_as_array = llvm::cast<llvm::ArrayType>(val_ty);
    llvm_val[stmt] = builder->CreateLoad(
        val_ty_as_array->getElementType(),
        builder->CreateGEP(physical_coordinate_ty, block_corner_coordinates,
                           {tlctx->get_constant(0), tlctx->get_constant(0), tlctx->get_constant(stmt->index)}));
  } else {
    QD_NOT_IMPLEMENTED;
  }
}

void TaskCodeGenLLVM::visit(GlobalTemporaryStmt *stmt) {
  auto runtime = get_runtime();
  auto buffer = call("get_temporary_pointer", runtime, tlctx->get_constant((int64)stmt->offset));

  auto ptr_type = llvm::PointerType::get(tlctx->get_data_type(stmt->ret_type.ptr_removed()), 0);
  llvm_val[stmt] = builder->CreatePointerCast(buffer, ptr_type);
}

void TaskCodeGenLLVM::visit(ThreadLocalPtrStmt *stmt) {
  auto base = get_tls_base_ptr();
  auto ptr = builder->CreateGEP(llvm::Type::getInt8Ty(*llvm_context), base, tlctx->get_constant(stmt->offset));
  auto ptr_type = llvm::PointerType::get(tlctx->get_data_type(stmt->ret_type.ptr_removed()), 0);
  llvm_val[stmt] = builder->CreatePointerCast(ptr, ptr_type);
}

void TaskCodeGenLLVM::visit(BlockLocalPtrStmt *stmt) {
  QD_ASSERT(bls_buffer);
  auto base = bls_buffer;
  auto ptr = builder->CreateGEP(base->getValueType(), base, {tlctx->get_constant(0), llvm_val[stmt->offset]});
  auto ptr_type = llvm::PointerType::get(tlctx->get_data_type(stmt->ret_type.ptr_removed()), 0);
  llvm_val[stmt] = builder->CreatePointerCast(ptr, ptr_type);
}

void TaskCodeGenLLVM::visit(ClearListStmt *stmt) {
  auto snode_child = stmt->snode;
  auto snode_parent = stmt->snode->parent;
  auto meta_child = cast_pointer(emit_struct_meta(snode_child), "StructMeta");
  auto meta_parent = cast_pointer(emit_struct_meta(snode_parent), "StructMeta");
  call("clear_list", get_runtime(), meta_parent, meta_child);
}

void TaskCodeGenLLVM::visit(InternalFuncStmt *stmt) {
  std::vector<llvm::Value *> args;

  if (stmt->with_runtime_context)
    args.push_back(get_context());

  for (auto s : stmt->args) {
    args.push_back(llvm_val[s]);
  }
  llvm_val[stmt] = call(stmt->func_name, std::move(args));
}

// Loads the per-kind split-heap base pointers from the runtime fields the launcher publishes (`_float` for f32 / f64
// allocas, `_int` for i32 / u1 allocas). Cached at `entry_block` so each downstream `AdStack*` visit reuses a
// dominating SSA value and `verifyFunction` stays happy regardless of which branch first triggered the load. Tasks
// with a captured `bound_expr` get the float heap sized to the reducer's gate-passing thread count; tasks without a
// captured gate fall back to the dispatched-threads worst case for the float heap. The int heap is always
// `num_threads * stride_int`.
void TaskCodeGenLLVM::ensure_ad_stack_heap_base_split_llvm() {
  if (ad_stack_heap_base_float_llvm_ != nullptr) {
    return;
  }
  llvm::IRBuilderBase::InsertPointGuard guard(*builder);
  builder->SetInsertPoint(entry_block);
  ad_stack_heap_base_float_llvm_ = call("LLVMRuntime_get_adstack_heap_buffer_float", get_runtime());
  ad_stack_heap_base_int_llvm_ = call("LLVMRuntime_get_adstack_heap_buffer_int", get_runtime());
}

// Cache the per-launch adstack metadata SSA values at `entry_block` on first need. Mirrors
// `ensure_ad_stack_heap_base_llvm`: one getter call per task, hoisted to the entry block so every downstream `AdStack*`
// visit (which may live in nested blocks) reuses a dominating SSA value and `verifyFunction` stays happy.
void TaskCodeGenLLVM::ensure_ad_stack_metadata_llvm() {
  if (ad_stack_stride_llvm_ != nullptr) {
    return;
  }
  llvm::IRBuilderBase::InsertPointGuard guard(*builder);
  builder->SetInsertPoint(entry_block);
  ad_stack_stride_llvm_ = call("LLVMRuntime_get_adstack_per_thread_stride", get_runtime());
  ad_stack_offsets_ptr_llvm_ = call("LLVMRuntime_get_adstack_offsets", get_runtime());
  ad_stack_max_sizes_ptr_llvm_ = call("LLVMRuntime_get_adstack_max_sizes", get_runtime());
}

// Split-heap counterpart that also loads the per-kind strides. `_float` drives the lazy float heap addressed by
// `row_id_var * stride_float + float_offset`; `_int` drives the eager int heap addressed by `linear_thread_idx *
// stride_int + int_offset`. Cached at `entry_block` like `ensure_ad_stack_metadata_llvm`. The legacy combined stride /
// offsets / max_sizes loads remain valid for tasks that have not migrated to the split layout.
void TaskCodeGenLLVM::ensure_ad_stack_metadata_split_llvm() {
  if (ad_stack_stride_float_llvm_ != nullptr) {
    return;
  }
  ensure_ad_stack_metadata_llvm();
  llvm::IRBuilderBase::InsertPointGuard guard(*builder);
  builder->SetInsertPoint(entry_block);
  ad_stack_stride_float_llvm_ = call("LLVMRuntime_get_adstack_per_thread_stride_float", get_runtime());
  ad_stack_stride_int_llvm_ = call("LLVMRuntime_get_adstack_per_thread_stride_int", get_runtime());
}

// Function-scope `alloca i32` holding the lazily-claimed float-heap row id for this task. Initialised to UINT32_MAX at
// task entry so any pre-LCA observation (none should reach a real read on a correct codegen) surfaces as an
// obviously-out-of-range index rather than aliasing row 0. The atomic-rmw claim at the float LCA block overwrites this
// with the per-thread row, after which every descendant float push / load-top reads the claimed value. The alloca is
// hoisted to the entry block (via the IRBuilder InsertPointGuard) regardless of where this helper is first called from,
// so `mem2reg` promotes it to SSA and the row id flows through downstream visits without per-site reloads.
llvm::Value *TaskCodeGenLLVM::ensure_ad_stack_row_id_var_float_llvm() {
  if (ad_stack_row_id_var_float_llvm_ != nullptr) {
    return ad_stack_row_id_var_float_llvm_;
  }
  llvm::IRBuilderBase::InsertPointGuard guard(*builder);
  builder->SetInsertPoint(entry_block, entry_block->getFirstInsertionPt());
  auto *i32ty = llvm::Type::getInt32Ty(*llvm_context);
  ad_stack_row_id_var_float_llvm_ = builder->CreateAlloca(i32ty);
  builder->CreateStore(llvm::ConstantInt::get(i32ty, std::numeric_limits<uint32_t>::max()),
                       ad_stack_row_id_var_float_llvm_);
  return ad_stack_row_id_var_float_llvm_;
}

// Emit the float-heap lazy row claim at the current insertion point. Called from `visit(Block *)` exactly once per task
// at the IR-level Lowest Common Ancestor (LCA) of every f32 push / load-top site (the same block the SPIR-V codegen
// pivots on at `spirv_codegen.cpp:visit(Block *)`):
//   - atomic-add 1 into `runtime->adstack_row_counters[task_codegen_id]` and read back the previous value
//   - clamp the claimed row against `runtime->adstack_bound_row_capacities[task_codegen_id]` so a reducer / main
//     divergence cannot OOB-write the heap; for tasks where the launcher did not publish a real capacity the slot holds
//     UINT32_MAX and the clamp is inert
//   - store the (possibly-clamped) row id into `ad_stack_row_id_var_float_llvm_` so every descendant float push /
//     load-top site reads it back
// Threads that never reach this block never claim a row and never touch the float heap, which is exactly the property
// the captured `bound_expr` reducer relies on to size the heap to gate-passing thread count.
void TaskCodeGenLLVM::emit_ad_stack_row_claim_llvm() {
  llvm::Value *row_id_var = ensure_ad_stack_row_id_var_float_llvm();

  auto *i32ty = llvm::Type::getInt32Ty(*llvm_context);
  auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
  llvm::Value *task_id_i64 = llvm::ConstantInt::get(i64ty, static_cast<uint64_t>(task_codegen_id));

  // Per-task counter slot: `runtime->adstack_row_counters[task_codegen_id]`.
  llvm::Value *row_counters_base = call("LLVMRuntime_get_adstack_row_counters", get_runtime());
  llvm::Value *counter_slot_ptr = builder->CreateGEP(i32ty, row_counters_base, task_id_i64);
  llvm::Value *one_i32 = llvm::ConstantInt::get(i32ty, 1);
  llvm::Value *claimed_row = builder->CreateAtomicRMW(llvm::AtomicRMWInst::Add, counter_slot_ptr, one_i32,
                                                      llvm::MaybeAlign(), llvm::AtomicOrdering::SequentiallyConsistent);

  // Per-task capacity slot for the defense-in-depth bounds check: clamp the claimed row at `capacity - 1` so any
  // overshoot stays in-bounds. For tasks without a captured `bound_expr` the launcher writes UINT32_MAX into this slot
  // so the clamp is inert. On overshoot (`claimed_row > capacity - 1`) the codegen also OR-1's the host-visible
  // adstack overflow flag (`runtime->adstack_overflow_flag_dev_ptr`, which the host allocated as pinned UVA-mapped
  // memory in `LlvmRuntimeExecutor::materialize_runtime`) so the host poll surfaces the divergence at the next
  // Quadrants Python entry. The atomic crosses the host/device boundary cleanly because the slot is in
  // pinned host memory; required hardware envelope is the same Pascal+ / GFX9+ that the existing pinned-host
  // H2D-async pattern already requires.
  llvm::Value *capacities_base = call("LLVMRuntime_get_adstack_bound_row_capacities", get_runtime());
  llvm::Value *capacity_slot_ptr = builder->CreateGEP(i32ty, capacities_base, task_id_i64);
  llvm::Value *capacity = builder->CreateLoad(i32ty, capacity_slot_ptr);
  // Guard the `capacity - 1` clamp upper bound against `capacity == 0`: a naive `capacity - 1` underflows to UINT32_MAX
  // and the clamp degenerates to a no-op, so any overshoot indexes off the heap end. Clamp the upper bound to row 0 in
  // that case (the launcher floors the heap allocation at one row precisely so this single-slot fallback is always
  // backed by real storage).
  llvm::Value *zero_i32 = llvm::ConstantInt::get(i32ty, 0);
  llvm::Value *capacity_is_zero = builder->CreateICmpEQ(capacity, zero_i32);
  llvm::Value *capacity_minus_one_raw = builder->CreateSub(capacity, one_i32);
  llvm::Value *clamp_upper = builder->CreateSelect(capacity_is_zero, zero_i32, capacity_minus_one_raw);
  llvm::Value *cmp = builder->CreateICmpUGT(claimed_row, clamp_upper);
  llvm::Value *clamped_row = builder->CreateSelect(cmp, clamp_upper, claimed_row);
  builder->CreateStore(clamped_row, row_id_var);

  // Overflow signal: on `claimed_row > clamp_upper`, atomically OR 1 into the pinned-host overflow flag and
  // record the offending task identity in the companion `adstack_overflow_task_id_dev_ptr` slot via a
  // `cmpxchg(0, registry_id)`. Only the FIRST overflowing thread's id sticks; subsequent threads observe
  // a non-zero value and their cmpxchg fails harmlessly. The condition is hoisted to a structured if so
  // the not-overflowing fast path skips both atomics entirely - one function call to fetch the pointers
  // plus one CreateICmpUGT comparison (the same compare we already emitted for the clamp).
  auto *current_function = builder->GetInsertBlock()->getParent();
  auto *overflow_then_block = llvm::BasicBlock::Create(*llvm_context, "adstack_overflow_signal", current_function);
  auto *overflow_merge_block = llvm::BasicBlock::Create(*llvm_context, "adstack_overflow_merge", current_function);
  builder->CreateCondBr(cmp, overflow_then_block, overflow_merge_block);
  builder->SetInsertPoint(overflow_then_block);
  {
    auto *i64ty_local = llvm::Type::getInt64Ty(*llvm_context);
    llvm::Value *flag_ptr = call("LLVMRuntime_get_adstack_overflow_flag_dev_ptr", get_runtime());
    llvm::Value *one_i64 = llvm::ConstantInt::get(i64ty_local, 1);
    builder->CreateAtomicRMW(llvm::AtomicRMWInst::Or, flag_ptr, one_i64, llvm::MaybeAlign(),
                             llvm::AtomicOrdering::Monotonic);
    // Record the registry id (0 means "not registered"; skip the cmpxchg in that case so the slot stays
    // zero and the host raise site falls through to the generic dual-cause message). Each offload task
    // emits its own lazy-claim block, so the immediate is task-local at codegen time.
    if (current_task != nullptr && current_task->ad_stack.registry_id != 0) {
      llvm::Value *task_id_ptr = call("LLVMRuntime_get_adstack_overflow_task_id_dev_ptr", get_runtime());
      llvm::Value *expected_zero = llvm::ConstantInt::get(i64ty_local, 0);
      llvm::Value *new_id =
          llvm::ConstantInt::get(i64ty_local, static_cast<uint64_t>(current_task->ad_stack.registry_id));
      builder->CreateAtomicCmpXchg(task_id_ptr, expected_zero, new_id, llvm::MaybeAlign(),
                                   llvm::AtomicOrdering::Monotonic, llvm::AtomicOrdering::Monotonic);
    }
    builder->CreateBr(overflow_merge_block);
  }
  builder->SetInsertPoint(overflow_merge_block);
}

// Return (creating on first call) the per-stack `alloca i64` that holds the live push count for this stack on the
// release-build path. The alloca is emitted in the entry block so `mem2reg` can promote it to an SSA register; the
// init-store of zero happens at the AdStackAllocaStmt visit site (which may sit inside a loop body, so each loop
// iteration that re-enters the AdStackAllocaStmt restarts the count - matching the `stack_init` semantics on the debug
// path).
llvm::Value *TaskCodeGenLLVM::ensure_ad_stack_count_alloca_llvm(const AdStackAllocaStmt *stack) {
  auto it = ad_stack_count_alloca_llvm_.find(stack);
  if (it != ad_stack_count_alloca_llvm_.end()) {
    return it->second;
  }
  llvm::IRBuilderBase::InsertPointGuard guard(*builder);
  builder->SetInsertPoint(entry_block, entry_block->getFirstInsertionPt());
  auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
  llvm::Value *count_alloca = builder->CreateAlloca(i64ty);
  ad_stack_count_alloca_llvm_[stack] = count_alloca;
  return count_alloca;
}

// True if the sizer has resolved this stack to a compile-time `max_size == 1` (a single-slot snapshot whose count
// is provably either 0 or 1 at every program point). The Const SizeExpr check rejects placeholder cases where
// `determine_ad_stack_size` set `max_size = 1` because the symbolic bound is non-Const and the runtime evaluates
// the actual capacity per launch. For these stacks the count alloca, mem2reg recurrence, and SCEV analysis are
// all dead - slot is always slot 0, push / loadtop / pop reduce to a constant-offset GEP.
static bool is_compile_time_single_slot(const AdStackAllocaStmt *stack) {
  return stack->max_size == 1 && stack->size_expr && stack->size_expr->kind == SizeExpr::Kind::Const &&
         stack->size_expr->const_value == 1;
}

// Constant-offset GEP into stack base for a single-slot stack. Slot index is fixed at 0, so the slot starts at
// byte offset `sizeof(u64)` (8) for the primal half and `sizeof(u64) + element_size` for the adjoint half.
llvm::Value *TaskCodeGenLLVM::emit_ad_stack_single_slot_ptr(const AdStackAllocaStmt *stack,
                                                            std::size_t adjoint_offset_bytes) {
  auto *i8ty = llvm::Type::getInt8Ty(*llvm_context);
  auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
  llvm::Value *slot_offset = llvm::ConstantInt::get(i64ty, sizeof(int64) + adjoint_offset_bytes);
  return builder->CreateGEP(i8ty, get_ad_stack_base_llvm(const_cast<AdStackAllocaStmt *>(stack)), slot_offset);
}

// Per-thread base pointer for the given alloca. Lazy float allocas (in tasks with a captured `bound_expr`) emit
// `heap_float + row_id_var * stride_float + offset` at every call site so the row claim from the LCA-block atomic-rmw
// is observed at each push / load-top rather than baked in at the alloca visit (which sees `row_id_var = UINT32_MAX`
// because it runs at the offload root, before the LCA). Every other alloca returns the cached base pointer set by
// `visit(AdStackAllocaStmt)`.
llvm::Value *TaskCodeGenLLVM::get_ad_stack_base_llvm(AdStackAllocaStmt *stack) {
  if (ad_stack_lazy_float_allocas_.count(stack) == 0) {
    return llvm_val[stack];
  }
  ensure_ad_stack_heap_base_split_llvm();
  ensure_ad_stack_metadata_split_llvm();
  llvm::Value *row_id_var = ensure_ad_stack_row_id_var_float_llvm();
  auto *i32ty = llvm::Type::getInt32Ty(*llvm_context);
  auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
  auto *i8ty = llvm::Type::getInt8Ty(*llvm_context);
  llvm::Value *row_id_i32 = builder->CreateLoad(i32ty, row_id_var);
  llvm::Value *row_id_i64 = builder->CreateZExt(row_id_i32, i64ty);
  llvm::Value *slice_offset = builder->CreateMul(row_id_i64, ad_stack_stride_float_llvm_);
  llvm::Value *stack_id_i64 = llvm::ConstantInt::get(i64ty, static_cast<uint64_t>(stack->stack_id));
  llvm::Value *offset_addr = builder->CreateGEP(i64ty, ad_stack_offsets_ptr_llvm_, stack_id_i64);
  llvm::Value *offset = builder->CreateLoad(i64ty, offset_addr);
  llvm::Value *total_offset = builder->CreateAdd(slice_offset, offset);
  return builder->CreateGEP(i8ty, ad_stack_heap_base_float_llvm_, total_offset);
}

// Compute the address of the top primal (or adjoint, when `adjoint_offset_bytes` == element_size) slot for an
// in-flight push count. Mirrors the runtime helper math `stack + sizeof(u64) + idx * 2 * element_size`, with `idx`
// being the saturating `count - 1` to match `stack_top_primal`'s underflow guard. Used by the release-build inline
// codegen for `AdStackPushStmt` / `AdStackLoadTopStmt` / `AdStackLoadTopAdjStmt` / `AdStackAccAdjointStmt`.
llvm::Value *TaskCodeGenLLVM::emit_ad_stack_top_slot_ptr(const AdStackAllocaStmt *stack,
                                                         llvm::Value *count,
                                                         std::size_t adjoint_offset_bytes) {
  auto *i8ty = llvm::Type::getInt8Ty(*llvm_context);
  auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
  llvm::Value *zero = llvm::ConstantInt::get(i64ty, 0);
  llvm::Value *one = llvm::ConstantInt::get(i64ty, 1);
  llvm::Value *count_minus_one = builder->CreateSub(count, one);
  llvm::Value *positive = builder->CreateICmpUGT(count, zero);
  llvm::Value *idx = builder->CreateSelect(positive, count_minus_one, zero);
  std::size_t entry_size = stack->entry_size_in_bytes();
  llvm::Value *slot_offset = builder->CreateAdd(llvm::ConstantInt::get(i64ty, sizeof(int64) + adjoint_offset_bytes),
                                                builder->CreateMul(idx, llvm::ConstantInt::get(i64ty, entry_size)));
  return builder->CreateGEP(i8ty, get_ad_stack_base_llvm(const_cast<AdStackAllocaStmt *>(stack)), slot_offset);
}

// Heap-backed adstack: the per-thread slice lives inside `runtime->adstack_heap_buffer`. The former
// `create_entry_block_alloca` path put the adstack on the worker-thread stack, which capped CPU reverse-mode
// kernels at the ~512 KB macOS secondary-thread budget and crashed with silently-zero gradients past that (the
// frame clobbered adjacent stack pages and downstream accumulators read zero). On CUDA / AMDGPU it put the
// adstack in per-thread local memory, which was not size-bounded but still duplicated across every kernel launch
// and burned per-thread register pressure. Heap-backing collapses both paths: one slab per runtime, sized
// `num_threads * per_thread_stride` by `LlvmRuntimeExecutor::ensure_adstack_heap` before each dispatch.
//
// The pre-scan in `init_offloaded_task_function` has already assigned this stmt a fixed `ad_stack_offsets_[stmt]`
// offset within the per-thread slice. At this visit site we compute `base = heap + thread_slot * stride + offset`
// and hand that pointer to `stack_init`, which writes the u64 count header exactly like the old stack-backed
// path. Downstream `stack_push/stack_pop/stack_top_primal/stack_top_adjoint` already take a raw `Ptr` and are
// unchanged - from their perspective the backing memory is still an opaque u64-prefixed blob.
void TaskCodeGenLLVM::visit(AdStackAllocaStmt *stmt) {
  QD_ASSERT_INFO(stmt->stack_id >= 0 && static_cast<std::size_t>(stmt->stack_id) < ad_stack_offsets_.size(),
                 "AdStackAllocaStmt reached visit without a pre-scanned stack_id - the scan in "
                 "init_offloaded_task_function must cover every container statement holding an adstack.");
  QD_ASSERT(ad_stack_per_thread_stride_ > 0);

  ensure_ad_stack_heap_base_split_llvm();
  ensure_ad_stack_metadata_split_llvm();

  // Unconditional split routing: float allocas address through `heap_float`, int / u1 allocas through `heap_int`,
  // regardless of whether the task captured a `bound_expr`. The two heaps are sized independently by the host launcher
  // (`ensure_adstack_heap_float` / `ensure_adstack_heap_int`); float can shrink to the reducer's count for bound_expr
  // tasks via `ensure_per_task_float_heap_post_reducer`, while int stays at `num_threads * stride_int`. Mirrors the
  // SPIR-V backend's unconditional `BufferType::AdStackHeapFloat` / `AdStackHeapInt` split.
  //
  // Float allocas in tasks with a captured `bound_expr` use the lazy claim path: do not bake a static base into
  // `llvm_val[stmt]` here because `linear_tid * stride` is the wrong index after the LCA-block atomic-rmw stores the
  // per-thread claimed row id into `ad_stack_row_id_var_float_llvm_`. Mark the alloca for `get_ad_stack_base_llvm` so
  // every push / load-top / load-top-adj / pop site recomputes the base as `heap_float + row_id_var * stride_float +
  // float_offset` at use time. Threads that never reach the LCA never claim a row and never reach a push / load-top by
  // definition of the LCA, so the unclaimed UINT32_MAX `row_id_var` is observed only at sites that do not execute.
  const bool is_float = stmt->ret_type == PrimitiveType::f32 || stmt->ret_type == PrimitiveType::f64;
  if (is_float && ad_stack_static_bound_expr_.has_value()) {
    ad_stack_lazy_float_allocas_.insert(stmt);
    if (compile_config.debug) {
      // Skip the `stack_init` call here when the alloca lives ABOVE the LCA block: `get_ad_stack_base_llvm(stmt)` would
      // emit `heap_float + row_id_var * stride_float + offset` while `row_id_var` is still its entry-block UINT32_MAX
      // init at this IR position (the LCA-block atomic-rmw row claim runs strictly later, after the gate IfStmt is
      // entered), and `stack_init`'s `*(u64*)stack = 0` would dereference that out-of-bounds address. The alloca's
      // matching stack_init is then emitted by the `visit(Block *)` LCA-block handler once the row claim has run.
      // When the alloca lives INSIDE the LCA block, by contrast, `visit(Block *)` has already emitted the row claim by
      // the time we get here - so `row_id_var` is valid and we can emit stack_init directly. Without this branch the
      // LCA-block handler would miss this alloca (its `for lazy_stmt : ad_stack_lazy_float_allocas_` iterates BEFORE
      // walking the block's statements, so the in-block alloca's insert above has not happened yet) and the heap u64
      // count header would never be explicitly zeroed - currently masked end-to-end by every backend's allocator
      // returning zeroed pages, but the contract "every lazy float alloca's stack_init runs before its first push"
      // should hold without relying on that. Initialise the per-stack count alloca either way, mirroring the release
      // path; the first `AdStackPushStmt` site under the LCA writes the `count` u64 header to its claimed row through
      // the same `stack_push` call that dereferences `row_id_var`.
      auto *i64ty_init = llvm::Type::getInt64Ty(*llvm_context);
      llvm::Value *count_alloca = ensure_ad_stack_count_alloca_llvm(stmt);
      builder->CreateStore(llvm::ConstantInt::get(i64ty_init, 0), count_alloca);
      if (stmt->parent != nullptr && stmt->parent == ad_stack_lca_block_float_ir_) {
        call("stack_init", get_ad_stack_base_llvm(stmt));
      }
      return;
    }
    if (is_compile_time_single_slot(stmt)) {
      return;
    }
    auto *i64ty_init = llvm::Type::getInt64Ty(*llvm_context);
    llvm::Value *count_alloca = ensure_ad_stack_count_alloca_llvm(stmt);
    builder->CreateStore(llvm::ConstantInt::get(i64ty_init, 0), count_alloca);
    return;
  }

  // Eager path for everything else: float allocas in non-bound_expr tasks address `heap_float + linear_tid *
  // stride_float + offset`; int allocas always address `heap_int + linear_tid * stride_int + offset`. Each alloca's
  // `host_offsets[stack_id]` is already an offset within its slice of the appropriate kind (float-only or int-only)
  // thanks to the host-side split publication in `publish_adstack_metadata`; we just pick the right base + stride pair
  // here.
  auto *i8ty = llvm::Type::getInt8Ty(*llvm_context);
  auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
  // Thread slot: on CPU it's `RuntimeContext::cpu_thread_id` (range [0, num_cpu_threads)); on CUDA / AMDGPU it's
  // `block_idx() * block_dim() + thread_idx()`. `linear_thread_idx(context)` is the runtime helper that returns the
  // arch-appropriate value, matching how `rand_states` is indexed and how the SPIR-V heap-backing indexes with
  // `gl_GlobalInvocationID`. Widen to u64 before the mul because a deep-AD kernel can easily cross `i32_max / stride`
  // on GPU grids (~65K threads x ~32K stride overflows i32).
  llvm::Value *linear_tid_i32 = call("linear_thread_idx", get_context());
  llvm::Value *linear_tid_i64 = builder->CreateZExt(linear_tid_i32, i64ty);
  llvm::Value *stride = is_float ? ad_stack_stride_float_llvm_ : ad_stack_stride_int_llvm_;
  llvm::Value *heap_base = is_float ? ad_stack_heap_base_float_llvm_ : ad_stack_heap_base_int_llvm_;
  llvm::Value *stack_id_i64 = llvm::ConstantInt::get(i64ty, static_cast<uint64_t>(stmt->stack_id));
  // `stride` and `offset` come from the per-launch metadata the host publishes via
  // `runtime_get_adstack_metadata_field_ptrs` rather than from codegen-time immediates. The old immediate path baked
  // the sum of compile-time `max_size` values into the kernel, which could not scale when a `SizeExpr` leaf resolved to
  // a different value at launch.
  llvm::Value *offset_addr = builder->CreateGEP(i64ty, ad_stack_offsets_ptr_llvm_, stack_id_i64);
  llvm::Value *offset = builder->CreateLoad(i64ty, offset_addr);
  llvm::Value *slice_offset = builder->CreateMul(linear_tid_i64, stride);
  llvm::Value *total_offset = builder->CreateAdd(slice_offset, offset);
  llvm::Value *stack_ptr = builder->CreateGEP(i8ty, heap_base, total_offset);
  llvm_val[stmt] = stack_ptr;
  if (compile_config.debug) {
    call("stack_init", llvm_val[stmt]);
    return;
  }
  if (is_compile_time_single_slot(stmt)) {
    // Single-slot specialization: count is provably either 0 (no push yet) or 1 (one push outstanding) at every
    // program point. Slot index is fixed at 0 so the slot pointer is a constant offset from the stack base.
    // Push / pop / loadtop reduce to constant-offset stores / loads with no count alloca, no mem2reg recurrence,
    // and no SCEV induction-variable analysis. Init is a no-op because no count state exists.
    return;
  }
  // Release build, multi-slot: store 0 into the per-stack count alloca instead of zeroing the heap u64 header.
  // Doing this at the AdStackAllocaStmt visit site (rather than once at task entry) restarts the count whenever
  // an outer loop re-enters the alloca, matching `stack_init`'s semantics on the bounds-checked path.
  auto *i64ty_init = llvm::Type::getInt64Ty(*llvm_context);
  llvm::Value *count_alloca = ensure_ad_stack_count_alloca_llvm(stmt);
  builder->CreateStore(llvm::ConstantInt::get(i64ty_init, 0), count_alloca);
}

void TaskCodeGenLLVM::visit(AdStackPopStmt *stmt) {
  if (compile_config.debug) {
    call("stack_pop", get_ad_stack_base_llvm(stmt->stack->as<AdStackAllocaStmt>()));
    return;
  }
  auto stack = stmt->stack->as<AdStackAllocaStmt>();
  if (is_compile_time_single_slot(stack)) {
    // Single-slot pop is a no-op: the next push (if any) overwrites slot 0 in place; the next loadtop reads
    // slot 0. There is no count state to decrement.
    return;
  }
  auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
  llvm::Value *count_alloca = ensure_ad_stack_count_alloca_llvm(stack);
  llvm::Value *count = builder->CreateLoad(i64ty, count_alloca);
  llvm::Value *zero = llvm::ConstantInt::get(i64ty, 0);
  llvm::Value *one = llvm::ConstantInt::get(i64ty, 1);
  llvm::Value *count_minus_one = builder->CreateSub(count, one);
  llvm::Value *positive = builder->CreateICmpUGT(count, zero);
  llvm::Value *new_count = builder->CreateSelect(positive, count_minus_one, zero);
  builder->CreateStore(new_count, count_alloca);
}

void TaskCodeGenLLVM::visit(AdStackPushStmt *stmt) {
  auto stack = stmt->stack->as<AdStackAllocaStmt>();
  // Autodiff-bootstrap const-init pushes (identified by the shared static-adstack analysis): keep the count_var
  // increment so the matching reverse pop balances, but skip the slot store. These pushes execute on every dispatched
  // thread regardless of any later gating; the bootstrap value is dead memory because no `load_top` ever reads it back.
  // Skipping the store is what lets the split-heap layout place the float row claim inside the gating branch without
  // dragging the LCA up to the offload root through these unconditional pushes; on the lazy float path the
  // runtime-helper `stack_push` (debug build) would otherwise dereference `heap_float + row_id_var * stride_float +
  // offset` while `row_id_var` is still its UINT32_MAX entry-block init at the bootstrap site (which sits ABOVE the LCA
  // where the atomic-rmw row claim writes the per-thread row id), and the count u64 store would land ~ TB past the heap
  // base. Same skip on debug as on release: the count_alloca increment alone keeps push and pop balanced, and the
  // bounds-check helper has nothing to do for an autodiff-emitted const-init that never reads back its slot anyway.
  if (ad_stack_bootstrap_pushes_.count(stmt) != 0) {
    // Single-slot adstacks have no `count_alloca` (the slot index is fixed at 0), so there is nothing to increment.
    // Multi-slot stacks bump `count_alloca` so the matching reverse pop balances. Either way we skip the slot store:
    // the bootstrap value is dead memory (no `load_top` ever reads it back) and the single-slot store would otherwise
    // route through `emit_ad_stack_single_slot_ptr -> get_ad_stack_base_llvm`, which on the lazy float path returns
    // `heap_float + row_id_var * stride_float + offset` while `row_id_var` is still its UINT32_MAX entry-block init at
    // the bootstrap site (the LCA-block atomic-rmw row claim runs strictly later) - the store would land ~ TB past the
    // heap base.
    if (!is_compile_time_single_slot(stack)) {
      auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
      llvm::Value *count_alloca = ensure_ad_stack_count_alloca_llvm(stack);
      llvm::Value *old_count = builder->CreateLoad(i64ty, count_alloca);
      llvm::Value *new_count = builder->CreateAdd(old_count, llvm::ConstantInt::get(i64ty, 1));
      builder->CreateStore(new_count, count_alloca);
    }
    return;
  }
  if (compile_config.debug) {
    // Debug build: route through the bounds-checking helper so any sizer bug surfaces as an overflow flag at sync. The
    // `max_size` load is only needed on this path.
    ensure_ad_stack_metadata_llvm();
    auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
    llvm::Value *stack_id_i64 = llvm::ConstantInt::get(i64ty, static_cast<uint64_t>(stack->stack_id));
    llvm::Value *max_size_addr = builder->CreateGEP(i64ty, ad_stack_max_sizes_ptr_llvm_, stack_id_i64);
    llvm::Value *max_size = builder->CreateLoad(i64ty, max_size_addr);
    llvm::Value *stack_base = get_ad_stack_base_llvm(stack);
    auto *i64ty_local = llvm::Type::getInt64Ty(*llvm_context);
    llvm::Value *registry_id_const = llvm::ConstantInt::get(
        i64ty_local, current_task != nullptr ? static_cast<uint64_t>(current_task->ad_stack.registry_id) : 0u);
    call("stack_push", get_runtime(), stack_base, max_size, tlctx->get_constant(stack->element_size_in_bytes()),
         registry_id_const);
    auto primal_ptr = call("stack_top_primal", stack_base, tlctx->get_constant(stack->element_size_in_bytes()));
    primal_ptr = builder->CreateBitCast(primal_ptr, llvm::PointerType::get(tlctx->get_data_type(stmt->ret_type), 0));
    builder->CreateStore(llvm_val[stmt->v], primal_ptr);
    return;
  }
  // Release build, multi-slot: emit the push as inline IR against the per-stack count alloca. After `mem2reg` promotes
  // the alloca to SSA, `GVN` folds the chain of `count++` across consecutive unrolled pushes; the only surviving memory
  // traffic in the unrolled body is the slot stores themselves. The runtime overflow check is dropped on this path
  // because `determine_ad_stack_size` produces a valid upper bound on per-thread push count along every execution path
  // (any unresolved stack is a hard compile error), so the `n + 1 > max_num_elements` guard inside `stack_push` is dead
  // in correct compilations. Single-slot stacks below skip the count alloca entirely - slot is fixed at offset 8.
  llvm::Value *primal_ptr;
  if (is_compile_time_single_slot(stack)) {
    primal_ptr = emit_ad_stack_single_slot_ptr(stack, /*adjoint_offset_bytes=*/0);
  } else {
    auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
    auto *i8ty = llvm::Type::getInt8Ty(*llvm_context);
    llvm::Value *count_alloca = ensure_ad_stack_count_alloca_llvm(stack);
    llvm::Value *old_count = builder->CreateLoad(i64ty, count_alloca);
    llvm::Value *new_count = builder->CreateAdd(old_count, llvm::ConstantInt::get(i64ty, 1));
    builder->CreateStore(new_count, count_alloca);
    // Slot is at index `new_count - 1`, which equals `old_count`. Skip the saturating-subtract that
    // `emit_ad_stack_top_slot_ptr` does because we just incremented and `new_count` is provably >= 1.
    std::size_t entry_size = stack->entry_size_in_bytes();
    llvm::Value *slot_offset =
        builder->CreateAdd(llvm::ConstantInt::get(i64ty, sizeof(int64)),
                           builder->CreateMul(old_count, llvm::ConstantInt::get(i64ty, entry_size)));
    primal_ptr = builder->CreateGEP(i8ty, get_ad_stack_base_llvm(stack), slot_offset);
  }
  // Zero the primal+adjoint slot pair to match `stack_push`'s `memset(top_primal, 0, 2 * element_size)`. Without this,
  // a previous use of this slot's adjoint would persist into the new push's accumulator. Slot pointer is `stack + 8 +
  // count * 2 * element_size` so the destination is `2 * element_size`-aligned (the slot stride), capped at 8 because
  // the per-thread slab base is 8-aligned. For `element_size in {1, 2}` (i8 / u1 packs, fp16) this is 2 or 4 bytes; an
  // over-stated alignment would let LLVM lower the memset to wider stores than the pointer can satisfy on stricter
  // backends.
  std::size_t slot_align = std::min<std::size_t>(8u, 2u * stack->element_size_in_bytes());
  builder->CreateMemSet(primal_ptr, llvm::ConstantInt::get(llvm::Type::getInt8Ty(*llvm_context), 0),
                        llvm::ConstantInt::get(llvm::Type::getInt64Ty(*llvm_context), stack->entry_size_in_bytes()),
                        llvm::MaybeAlign(slot_align));
  llvm::Value *primal_typed_ptr =
      builder->CreateBitCast(primal_ptr, llvm::PointerType::get(tlctx->get_data_type(stmt->ret_type), 0));
  builder->CreateStore(llvm_val[stmt->v], primal_typed_ptr);
}

void TaskCodeGenLLVM::visit(AdStackLoadTopStmt *stmt) {
  QD_ASSERT(stmt->return_ptr == false);
  auto stack = stmt->stack->as<AdStackAllocaStmt>();
  if (compile_config.debug) {
    auto primal_ptr =
        call("stack_top_primal", get_ad_stack_base_llvm(stack), tlctx->get_constant(stack->element_size_in_bytes()));
    auto primal_ty = tlctx->get_data_type(stmt->ret_type);
    primal_ptr = builder->CreateBitCast(primal_ptr, llvm::PointerType::get(primal_ty, 0));
    llvm_val[stmt] = builder->CreateLoad(primal_ty, primal_ptr);
    return;
  }
  llvm::Value *primal_ptr;
  if (is_compile_time_single_slot(stack)) {
    primal_ptr = emit_ad_stack_single_slot_ptr(stack, /*adjoint_offset_bytes=*/0);
  } else {
    auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
    llvm::Value *count_alloca = ensure_ad_stack_count_alloca_llvm(stack);
    llvm::Value *count = builder->CreateLoad(i64ty, count_alloca);
    primal_ptr = emit_ad_stack_top_slot_ptr(stack, count, /*adjoint_offset_bytes=*/0);
  }
  auto primal_ty = tlctx->get_data_type(stmt->ret_type);
  primal_ptr = builder->CreateBitCast(primal_ptr, llvm::PointerType::get(primal_ty, 0));
  llvm_val[stmt] = builder->CreateLoad(primal_ty, primal_ptr);
}

void TaskCodeGenLLVM::visit(AdStackLoadTopAdjStmt *stmt) {
  auto stack = stmt->stack->as<AdStackAllocaStmt>();
  if (compile_config.debug) {
    auto adjoint =
        call("stack_top_adjoint", get_ad_stack_base_llvm(stack), tlctx->get_constant(stack->element_size_in_bytes()));
    auto adjoint_ty = tlctx->get_data_type(stmt->ret_type);
    adjoint = builder->CreateBitCast(adjoint, llvm::PointerType::get(adjoint_ty, 0));
    llvm_val[stmt] = builder->CreateLoad(adjoint_ty, adjoint);
    return;
  }
  llvm::Value *adjoint_ptr;
  if (is_compile_time_single_slot(stack)) {
    adjoint_ptr = emit_ad_stack_single_slot_ptr(stack, /*adjoint_offset_bytes=*/stack->element_size_in_bytes());
  } else {
    auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
    llvm::Value *count_alloca = ensure_ad_stack_count_alloca_llvm(stack);
    llvm::Value *count = builder->CreateLoad(i64ty, count_alloca);
    adjoint_ptr = emit_ad_stack_top_slot_ptr(stack, count, /*adjoint_offset_bytes=*/stack->element_size_in_bytes());
  }
  auto adjoint_ty = tlctx->get_data_type(stmt->ret_type);
  adjoint_ptr = builder->CreateBitCast(adjoint_ptr, llvm::PointerType::get(adjoint_ty, 0));
  llvm_val[stmt] = builder->CreateLoad(adjoint_ty, adjoint_ptr);
}

void TaskCodeGenLLVM::visit(AdStackAccAdjointStmt *stmt) {
  auto stack = stmt->stack->as<AdStackAllocaStmt>();
  llvm::Value *adjoint_ptr;
  if (compile_config.debug) {
    adjoint_ptr =
        call("stack_top_adjoint", get_ad_stack_base_llvm(stack), tlctx->get_constant(stack->element_size_in_bytes()));
  } else if (is_compile_time_single_slot(stack)) {
    adjoint_ptr = emit_ad_stack_single_slot_ptr(stack, /*adjoint_offset_bytes=*/stack->element_size_in_bytes());
  } else {
    auto *i64ty = llvm::Type::getInt64Ty(*llvm_context);
    llvm::Value *count_alloca = ensure_ad_stack_count_alloca_llvm(stack);
    llvm::Value *count = builder->CreateLoad(i64ty, count_alloca);
    adjoint_ptr = emit_ad_stack_top_slot_ptr(stack, count, /*adjoint_offset_bytes=*/stack->element_size_in_bytes());
  }
  auto adjoint_ty = tlctx->get_data_type(stack->ret_type);
  adjoint_ptr = builder->CreateBitCast(adjoint_ptr, llvm::PointerType::get(adjoint_ty, 0));
  auto old_val = builder->CreateLoad(adjoint_ty, adjoint_ptr);
  QD_ASSERT(is_real(stmt->v->ret_type));
  auto new_val = builder->CreateFAdd(old_val, llvm_val[stmt->v]);
  builder->CreateStore(new_val, adjoint_ptr);
}

void TaskCodeGenLLVM::visit(RangeAssumptionStmt *stmt) {
  llvm_val[stmt] = llvm_val[stmt->input];
}

void TaskCodeGenLLVM::visit(LoopUniqueStmt *stmt) {
  llvm_val[stmt] = llvm_val[stmt->input];
}

void TaskCodeGenLLVM::visit_call_bitcode(ExternalFuncCallStmt *stmt) {
  QD_ASSERT(stmt->type == ExternalFuncCallStmt::BITCODE);
  std::vector<llvm::Value *> arg_values;
  for (const auto &s : stmt->arg_stmts)
    arg_values.push_back(llvm_val[s]);
  // Link external module to the core module
  if (linked_modules.find(stmt->bc_filename) == linked_modules.end()) {
    linked_modules.insert(stmt->bc_filename);
    std::unique_ptr<llvm::Module> external_module = module_from_bitcode_file(stmt->bc_filename, llvm_context);
    auto *func_ptr = external_module->getFunction(stmt->bc_funcname);
    QD_ASSERT_INFO(func_ptr != nullptr, "{} is not found in {}.", stmt->bc_funcname, stmt->bc_filename);
    auto link_error = llvm::Linker::linkModules(*module, std::move(external_module));
    QD_ASSERT(!link_error);
  }
  // Retrieve function again. Do it here to detect name conflicting.
  auto *func_ptr = module->getFunction(stmt->bc_funcname);
  // Convert pointer type from a[n * m] to a[n][m]
  for (int i = 0; i < func_ptr->getFunctionType()->getNumParams(); ++i) {
    QD_ASSERT_INFO(func_ptr->getArg(i)->getType()->getTypeID() == arg_values[i]->getType()->getTypeID(),
                   "TypeID {} != {} with {}", (int)func_ptr->getArg(i)->getType()->getTypeID(),
                   (int)arg_values[i]->getType()->getTypeID(), i);
    auto tmp_value = arg_values[i];
    arg_values[i] = builder->CreatePointerCast(tmp_value, func_ptr->getArg(i)->getType());
  }
  call(func_ptr, arg_values);
}

void TaskCodeGenLLVM::visit_call_shared_object(ExternalFuncCallStmt *stmt) {
  QD_ASSERT(stmt->type == ExternalFuncCallStmt::SHARED_OBJECT);
  std::vector<llvm::Type *> arg_types;
  std::vector<llvm::Value *> arg_values;

  for (const auto &s : stmt->arg_stmts) {
    arg_types.push_back(tlctx->get_data_type(s->ret_type));
    arg_values.push_back(llvm_val[s]);
  }

  for (const auto &s : stmt->output_stmts) {
    auto t = tlctx->get_data_type(s->ret_type);
    auto ptr = llvm::PointerType::get(t, 0);
    arg_types.push_back(ptr);
    arg_values.push_back(llvm_val[s]);
  }

  auto func_type = llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_context), arg_types, false);
  auto func_ptr_type = llvm::PointerType::get(func_type, 0);

  auto addr = tlctx->get_constant((std::size_t)stmt->so_func);
  auto func = builder->CreateIntToPtr(addr, func_ptr_type);
  call(func, func_type, arg_values);
}

void TaskCodeGenLLVM::visit(ExternalFuncCallStmt *stmt) {
  QD_NOT_IMPLEMENTED
}

void TaskCodeGenLLVM::visit(MeshPatchIndexStmt *stmt) {
  llvm_val[stmt] = get_arg(2);
}

void TaskCodeGenLLVM::visit(MatrixInitStmt *stmt) {
  auto type = tlctx->get_data_type(stmt->ret_type->as<TensorType>());
  llvm::Value *vec = llvm::UndefValue::get(type);
  for (int i = 0; i < stmt->values.size(); ++i) {
    auto *elem = llvm_val[stmt->values[i]];
    if (codegen_vector_type(compile_config)) {
      QD_ASSERT(llvm::dyn_cast<llvm::VectorType>(type));
      vec = builder->CreateInsertElement(vec, elem, i);
    } else {
      QD_ASSERT(llvm::dyn_cast<llvm::ArrayType>(type));
      vec = builder->CreateInsertValue(vec, elem, i);
    }
  }
  llvm_val[stmt] = vec;
}

void TaskCodeGenLLVM::eliminate_unused_functions() {
  QuadrantsLLVMContext::eliminate_unused_functions(module.get(), [&](std::string func_name) {
    for (auto &task : offloaded_tasks) {
      if (task.name == func_name)
        return true;
    }
    return false;
  });
}

FunctionCreationGuard TaskCodeGenLLVM::get_function_creation_guard(std::vector<llvm::Type *> argument_types,
                                                                   const std::string &func_name) {
  return FunctionCreationGuard(this, argument_types, func_name);
}

void TaskCodeGenLLVM::initialize_context() {
  QD_ASSERT(tlctx != nullptr);
  llvm_context = tlctx->get_this_thread_context();
  builder = std::make_unique<llvm::IRBuilder<>>(*llvm_context);
  if (compile_config.fast_math) {
    llvm::FastMathFlags fast_flags;
    fast_flags.setNoInfs();
    fast_flags.setNoSignedZeros();
    fast_flags.setAllowReassoc();
    fast_flags.setApproxFunc();
    builder->setFastMathFlags(fast_flags);
  }
}

llvm::Value *TaskCodeGenLLVM::get_arg(int i) {
  std::vector<llvm::Value *> args;
  for (auto &arg : func->args()) {
    args.push_back(&arg);
  }
  return args[i];
}

llvm::Value *TaskCodeGenLLVM::get_context() {
  return get_arg(0);
}

llvm::Value *TaskCodeGenLLVM::get_tls_base_ptr() {
  return get_arg(1);
}

llvm::Type *TaskCodeGenLLVM::get_tls_buffer_type() {
  return llvm::PointerType::getUnqual(*llvm_context);
}

std::vector<llvm::Type *> TaskCodeGenLLVM::get_xlogue_argument_types() {
  return {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0), get_tls_buffer_type()};
}

std::vector<llvm::Type *> TaskCodeGenLLVM::get_mesh_xlogue_argument_types() {
  return {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0), get_tls_buffer_type(),
          tlctx->get_data_type<uint32_t>()};
}

llvm::Type *TaskCodeGenLLVM::get_xlogue_function_type() {
  return llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_context), get_xlogue_argument_types(), false);
}

llvm::Type *TaskCodeGenLLVM::get_mesh_xlogue_function_type() {
  return llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_context), get_mesh_xlogue_argument_types(), false);
}

llvm::IntegerType *TaskCodeGenLLVM::get_integer_type(int bits) {
  switch (bits) {
    case 8:
      return llvm::Type::getInt8Ty(*llvm_context);
    case 16:
      return llvm::Type::getInt16Ty(*llvm_context);
    case 32:
      return llvm::Type::getInt32Ty(*llvm_context);
    case 64:
      return llvm::Type::getInt64Ty(*llvm_context);
    default:
      break;
  }
  QD_ERROR("No compatible " + std::to_string(bits) + " bits integer type.");
  return nullptr;
}

llvm::Value *TaskCodeGenLLVM::get_root(int snode_tree_id) {
  return call("LLVMRuntime_get_roots", get_runtime(), tlctx->get_constant(snode_tree_id));
}

llvm::Value *TaskCodeGenLLVM::get_runtime() {
  auto runtime_ptr = call("RuntimeContext_get_runtime", get_context());
  return builder->CreateBitCast(runtime_ptr, llvm::PointerType::get(get_runtime_type("LLVMRuntime"), 0));
}

llvm::Value *TaskCodeGenLLVM::emit_struct_meta(SNode *snode) {
  auto obj = emit_struct_meta_object(snode);
  QD_ASSERT(obj != nullptr);
  return obj->ptr;
}

void TaskCodeGenLLVM::emit_to_module() {
  QD_AUTO_PROF
  ir->accept(this);
}

LLVMCompiledTask TaskCodeGenLLVM::run_compilation() {
  // Final lowering
  auto offload_to_executable = [](IRNode *ir, const CompileConfig &config, const Kernel *kernel) {
    bool verbose = config.print_ir;
    if (kernel->is_accessor && !config.print_accessor_ir) {
      verbose = false;
    }
    irpass::offload_to_executable(ir, config, kernel, verbose,
                                  /*determine_ad_stack_size=*/kernel->autodiff_mode == AutodiffMode::kReverse,
                                  /*lower_global_access=*/true,
                                  /*make_thread_local=*/config.make_thread_local,
                                  /*make_block_local=*/
                                  is_extension_supported(config.arch, Extension::bls) && config.make_block_local);
  };

  offload_to_executable(ir, compile_config, kernel);

  emit_to_module();
  eliminate_unused_functions();

  if (compile_config.arch == Arch::cuda) {
    // CUDA specific metadata
    for (const auto &task : offloaded_tasks) {
      llvm::Function *func = module->getFunction(task.name);
      QD_ASSERT(func);
      tlctx->mark_function_as_cuda_kernel(func, task.block_dim);
    }
  } else if (compile_config.arch == Arch::amdgpu) {
    for (const auto &task : offloaded_tasks) {
      llvm::Function *func = module->getFunction(task.name);
      QD_ASSERT(func);
      tlctx->mark_function_as_amdgpu_kernel(func);
    }
#if defined(QD_WITH_AMDGPU)
    llvm::legacy::FunctionPassManager fpm(module.get());
    fpm.add(new AMDGPUConvertAllocaInstAddressSpacePass());
    fpm.doInitialization();
    for (auto &func : *module)
      fpm.run(func);
    fpm.doFinalization();
#endif
  }
  std::filesystem::path ir_dump_dir = compile_config.debug_dump_path;
  if (get_environ_config(DUMP_IR_ENV.data())) {
    std::filesystem::create_directories(ir_dump_dir);

    QD_ASSERT(!offloaded_tasks.empty());
    std::string dump_name = offloaded_tasks[0].name;
    std::filesystem::path filename = ir_dump_dir / (dump_name + "_llvm.ll");
    std::error_code EC;
    llvm::raw_fd_ostream dest_file(filename.string(), EC);
    if (!EC) {
      module->print(dest_file, nullptr);
    }
  }

  if (get_environ_config(LOAD_IR_ENV.data())) {
    QD_ASSERT(!offloaded_tasks.empty());
    std::filesystem::path filename = ir_dump_dir / (offloaded_tasks[0].name + "_llvm.ll");
    llvm::SMDiagnostic err;
    auto loaded_module = llvm::parseAssemblyFile(filename.string(), err, *llvm_context);
    if (!loaded_module) {
      err.print("QUADRANTS_LOAD_IR_FILE error", llvm::errs());
      QD_ERROR("Failed to load LLVM IR from {}", filename.string());
    } else {
      module = std::move(loaded_module);
    }
  }

  return {std::move(offloaded_tasks), std::move(module), std::move(used_tree_ids), std::move(struct_for_tls_sizes)};
}

llvm::Value *TaskCodeGenLLVM::create_xlogue(std::unique_ptr<Block> &block) {
  llvm::Value *xlogue;

  auto xlogue_type = get_xlogue_function_type();
  auto xlogue_ptr_type = llvm::PointerType::get(xlogue_type, 0);

  if (block) {
    auto guard = get_function_creation_guard(get_xlogue_argument_types());
    block->accept(this);
    xlogue = guard.body;
  } else {
    xlogue = llvm::ConstantPointerNull::get(xlogue_ptr_type);
  }

  return xlogue;
}

llvm::Value *TaskCodeGenLLVM::create_mesh_xlogue(std::unique_ptr<Block> &block) {
  llvm::Value *xlogue;

  auto xlogue_type = get_mesh_xlogue_function_type();
  auto xlogue_ptr_type = llvm::PointerType::get(xlogue_type, 0);

  if (block) {
    auto guard = get_function_creation_guard(get_mesh_xlogue_argument_types());
    block->accept(this);
    xlogue = guard.body;
  } else {
    xlogue = llvm::ConstantPointerNull::get(xlogue_ptr_type);
  }

  return xlogue;
}

void TaskCodeGenLLVM::visit(ReferenceStmt *stmt) {
  llvm_val[stmt] = llvm_val[stmt->var];
}

void TaskCodeGenLLVM::visit(FuncCallStmt *stmt) {
  if (!func_map.count(stmt->func)) {
    auto guard = get_function_creation_guard({llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0)},
                                             stmt->func->get_name());
    const Callable *old_callable = current_callable;
    current_callable = stmt->func;
    func_map.insert({stmt->func, guard.body});
    stmt->func->ir->accept(this);
    current_callable = old_callable;
  }
  llvm::Function *llvm_func = func_map[stmt->func];
  // FIXME: when cpu_assert_failed fires inside a @qd.real_func callee, the
  // flag is set on new_ctx but never propagated back to the caller's context.
  // Regular @qd.func is AST-inlined so assertions are handled by the caller's
  // visit(AssertStmt) directly.  real_func needs: (1) zero-init new_ctx's
  // cpu_assert_failed before the call, (2) post-call check + propagate to
  // get_context(), (3) emit ret void on failure.
  auto *new_ctx = create_entry_block_alloca(get_runtime_type("RuntimeContext"));
  call("RuntimeContext_set_runtime", new_ctx, get_runtime());
  if (!stmt->func->parameter_list.empty()) {
    auto *buffer = create_entry_block_alloca(tlctx->get_data_type(stmt->func->args_type));
    set_args_ptr(stmt->func, new_ctx, buffer);
    set_struct_to_buffer(stmt->func->args_type, buffer, stmt->args);
  }
  llvm::Value *result_buffer = nullptr;
  if (!stmt->func->rets.empty()) {
    auto *ret_type = tlctx->get_data_type(stmt->func->ret_type);
    result_buffer = create_entry_block_alloca(ret_type);
    auto *result_buffer_u64 =
        builder->CreatePointerCast(result_buffer, llvm::PointerType::get(tlctx->get_data_type<uint64>(), 0));
    call("RuntimeContext_set_result_buffer", new_ctx, result_buffer_u64);
  }
  call(llvm_func, new_ctx);
  llvm_val[stmt] = result_buffer;
}

void TaskCodeGenLLVM::visit(GetElementStmt *stmt) {
  auto *struct_type = tlctx->get_data_type(stmt->src->ret_type.ptr_removed());
  std::vector<llvm::Value *> index;
  index.reserve(stmt->index.size() + 1);
  index.push_back(tlctx->get_constant(0));
  for (auto &i : stmt->index) {
    index.push_back(tlctx->get_constant(i));
  }
  auto *gep = builder->CreateGEP(struct_type, llvm_val[stmt->src], index);
  auto *val = builder->CreateLoad(tlctx->get_data_type(stmt->ret_type), gep);
  llvm_val[stmt] = val;
}

void TaskCodeGenLLVM::set_struct_to_buffer(llvm::Value *buffer,
                                           llvm::Type *buffer_type,
                                           const std::vector<Stmt *> &elements,
                                           const Type *current_type,
                                           int &current_element,
                                           std::vector<llvm::Value *> &current_index) {
  if (auto primitive_type = current_type->cast<PrimitiveType>()) {
    QD_ASSERT((Type *)elements[current_element]->ret_type == current_type);
    auto *gep = builder->CreateGEP(buffer_type, buffer, current_index);
    builder->CreateStore(llvm_val[elements[current_element]], gep);
    current_element++;
  } else if (auto pointer_type = current_type->cast<PointerType>()) {
    QD_ASSERT((Type *)elements[current_element]->ret_type == current_type);
    auto *gep = builder->CreateGEP(buffer_type, buffer, current_index);
    builder->CreateStore(llvm_val[elements[current_element]], gep);
    current_element++;
  } else if (auto struct_type = current_type->cast<StructType>()) {
    int i = 0;
    for (const auto &element : struct_type->elements()) {
      current_index.push_back(tlctx->get_constant(i++));
      set_struct_to_buffer(buffer, buffer_type, elements, element.type, current_element, current_index);
      current_index.pop_back();
    }
  } else if (auto tensor_type = current_type->cast<TensorType>()) {
    int num_elements = tensor_type->get_num_elements();
    Type *element_type = tensor_type->get_element_type();
    for (int i = 0; i < num_elements; i++) {
      current_index.push_back(tlctx->get_constant(i));
      set_struct_to_buffer(buffer, buffer_type, elements, element_type, current_element, current_index);
      current_index.pop_back();
    }
  } else {
    QD_INFO("{}", current_type->to_string());
    QD_NOT_IMPLEMENTED
  }
}

void TaskCodeGenLLVM::set_struct_to_buffer(const StructType *struct_type,
                                           llvm::Value *buffer,
                                           const std::vector<Stmt *> &elements) {
  auto buffer_type = tlctx->get_data_type(struct_type);
  buffer = builder->CreatePointerCast(buffer, llvm::PointerType::get(buffer_type, 0));
  int current_element = 0;
  std::vector<llvm::Value *> current_index = {tlctx->get_constant(0)};
  set_struct_to_buffer(buffer, buffer_type, elements, struct_type, current_element, current_index);
}

llvm::Value *TaskCodeGenLLVM::get_struct_arg(const std::vector<int> &index, bool create_load) {
  auto *args_ptr = get_args_ptr(current_callable, get_context());
  auto *args_type = current_callable->args_type;
  auto *arg_type = args_type->get_element_type(index);
  std::vector<llvm::Value *> gep_index;
  gep_index.reserve(index.size() + 1);
  gep_index.push_back(tlctx->get_constant(0));
  for (int ind : index) {
    gep_index.push_back(tlctx->get_constant(ind));
  }
  auto *gep = builder->CreateGEP(tlctx->get_data_type(args_type), args_ptr, gep_index);
  if (!create_load) {
    return gep;
  }
  return builder->CreateLoad(tlctx->get_data_type(arg_type), gep);
}

llvm::Value *TaskCodeGenLLVM::get_args_ptr(const Callable *callable, llvm::Value *context) {
  auto *runtime_context_type = get_runtime_type("RuntimeContext");
  auto *args_type = tlctx->get_data_type(callable->args_type);
  auto *zero = tlctx->get_constant(0);
  // The address of the arg buffer is the first element of RuntimeContext
  auto *args_ptr = builder->CreateGEP(runtime_context_type, context, {zero, zero});
  // casting from i8 ** to args_type **
  args_ptr = builder->CreatePointerCast(args_ptr, llvm::PointerType::get(llvm::PointerType::get(args_type, 0), 0));
  // loading the address of the arg buffer (args_type *)
  args_ptr = builder->CreateLoad(llvm::PointerType::get(args_type, 0), args_ptr);
  return args_ptr;
}
void TaskCodeGenLLVM::set_args_ptr(Callable *callable, llvm::Value *context, llvm::Value *ptr) {
  auto *runtime_context_type = get_runtime_type("RuntimeContext");
  auto *args_type = tlctx->get_data_type(callable->args_type);
  auto *zero = tlctx->get_constant(0);
  // The address of the arg buffer is the first element of RuntimeContext
  auto *args_ptr = builder->CreateGEP(runtime_context_type, context, {zero, zero});
  // casting from i8 ** to args_type **
  args_ptr = builder->CreatePointerCast(args_ptr, llvm::PointerType::get(llvm::PointerType::get(args_type, 0), 0));
  // storing the address of the arg buffer (args_type *)
  builder->CreateStore(ptr, args_ptr);
};

LLVMCompiledTask LLVMCompiledTask::clone() const {
  return {tasks, llvm::CloneModule(*module), used_tree_ids, struct_for_tls_sizes};
}

LLVMCompiledKernel LLVMCompiledKernel::clone() const {
  return {tasks, llvm::CloneModule(*module)};
}

}  // namespace quadrants::lang

#endif  // #ifdef QD_WITH_LLVM
