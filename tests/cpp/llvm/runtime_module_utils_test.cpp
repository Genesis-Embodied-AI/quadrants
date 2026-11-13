#include "gtest/gtest.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "quadrants/runtime/llvm/runtime_module_utils.h"

namespace quadrants::lang {
namespace {

TEST(RuntimeModuleUtils, ForwardsArgumentsAndAppendsOverloadedIntrinsicArguments) {
  llvm::LLVMContext context;
  llvm::Module module("runtime", context);
  llvm::IRBuilder<> builder(context);
  auto *i32 = builder.getInt32Ty();
  auto *function = llvm::Function::Create(llvm::FunctionType::get(i32, {i32}, false),
                                          llvm::GlobalValue::ExternalLinkage, "count_zeros", module);
  builder.SetInsertPoint(llvm::BasicBlock::Create(context, "old", function));
  builder.CreateRet(builder.getInt32(99));
  function->addFnAttr(llvm::Attribute::OptimizeNone);
  function->addFnAttr(llvm::Attribute::NoInline);

  patch_runtime_intrinsic(module, "count_zeros", llvm::Intrinsic::cttz, true, {i32}, {builder.getFalse()});

  ASSERT_EQ(function->size(), 1);
  auto *call = llvm::dyn_cast<llvm::CallInst>(&function->front().front());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->getCalledFunction()->getIntrinsicID(), llvm::Intrinsic::cttz);
  ASSERT_EQ(call->arg_size(), 2);
  EXPECT_EQ(call->getArgOperand(0), function->getArg(0));
  EXPECT_EQ(call->getArgOperand(1), builder.getFalse());
  auto *ret = llvm::cast<llvm::ReturnInst>(function->front().getTerminator());
  EXPECT_EQ(ret->getReturnValue(), call);
  EXPECT_TRUE(function->hasFnAttribute(llvm::Attribute::AlwaysInline));
  EXPECT_FALSE(function->hasFnAttribute(llvm::Attribute::OptimizeNone));
  EXPECT_FALSE(function->hasFnAttribute(llvm::Attribute::NoInline));
  EXPECT_FALSE(llvm::verifyModule(module, &llvm::errs()));
}

TEST(RuntimeModuleUtils, SuppliesExtraArgumentToVoidCudaIntrinsic) {
  llvm::LLVMContext context;
  llvm::Module module("runtime", context);
  llvm::IRBuilder<> builder(context);
  auto *function = llvm::Function::Create(llvm::FunctionType::get(builder.getVoidTy(), false),
                                          llvm::GlobalValue::ExternalLinkage, "block_barrier", module);

  patch_runtime_intrinsic(module, "block_barrier", llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all, false, {},
                          {builder.getInt32(0)});

  auto *call = llvm::cast<llvm::CallInst>(&function->front().front());
  ASSERT_EQ(call->arg_size(), 1);
  EXPECT_EQ(call->getArgOperand(0), builder.getInt32(0));
  EXPECT_EQ(call->getCalledFunction()->getIntrinsicID(), llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all);
  EXPECT_EQ(llvm::cast<llvm::ReturnInst>(function->front().getTerminator())->getReturnValue(), nullptr);
  EXPECT_FALSE(llvm::verifyModule(module, &llvm::errs()));
}

TEST(RuntimeModuleUtils, MissingStubDoesNotAddAnIntrinsicDeclaration) {
  llvm::LLVMContext context;
  llvm::Module module("runtime", context);
  patch_runtime_intrinsic(module, "absent", llvm::Intrinsic::cttz, true, {llvm::Type::getInt32Ty(context)});
  EXPECT_TRUE(module.empty());
}

TEST(RuntimeModuleUtils, PreservesExplicitNoInlineMarker) {
  llvm::LLVMContext context;
  llvm::Module module("runtime", context);
  llvm::IRBuilder<> builder(context);
  auto *type = llvm::FunctionType::get(builder.getVoidTy(), false);
  auto *marker = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage, "mark_force_no_inline", module);
  auto *function = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage, "caller", module);
  builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", function));
  builder.CreateCall(marker);
  builder.CreateRetVoid();
  function->addFnAttr(llvm::Attribute::OptimizeNone);
  function->addFnAttr(llvm::Attribute::NoInline);

  mark_runtime_function_inline(function);

  EXPECT_TRUE(function->hasFnAttribute(llvm::Attribute::OptimizeNone));
  EXPECT_TRUE(function->hasFnAttribute(llvm::Attribute::NoInline));
  EXPECT_FALSE(function->hasFnAttribute(llvm::Attribute::AlwaysInline));
  EXPECT_FALSE(llvm::verifyModule(module, &llvm::errs()));
}

}  // namespace
}  // namespace quadrants::lang
