#include "gtest/gtest.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "quadrants/runtime/amdgpu/runtime_hsaco_cache_key.h"

namespace quadrants::lang {
namespace {

TEST(RuntimeHsacoCache, KeyTracksModuleContents) {
  llvm::LLVMContext context;
  llvm::Module module("runtime", context);
  auto type = llvm::FunctionType::get(llvm::Type::getInt32Ty(context), false);
  auto function = llvm::Function::Create(type, llvm::Function::ExternalLinkage, "runtime_test", module);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
  auto ret = builder.CreateRet(builder.getInt32(1));
  auto key = [&] { return make_runtime_hsaco_key(module, "gfx900", "+wavefrontsize64", true); };
  const auto original = key();
  EXPECT_EQ(original, key());
  ret->setOperand(0, builder.getInt32(2));
  EXPECT_FALSE(original == key());
  ret->setOperand(0, builder.getInt32(1));
  EXPECT_EQ(original, key());
  function->addFnAttr("target-features", "+wavefrontsize32");
  EXPECT_FALSE(original == key());
  function->removeFnAttr("target-features");
  EXPECT_EQ(original, key());
  module.setDataLayout("e-p:64:64");
  EXPECT_FALSE(original == key());
  auto with_layout = key();
  module.setTargetTriple(llvm::Triple("amdgcn-amd-amdhsa"));
  EXPECT_FALSE(with_layout == key());
}

TEST(RuntimeHsacoCache, KeyIsIndependentOfLLVMContextLifetime) {
  auto make_key = [] {
    llvm::LLVMContext context;
    llvm::Module module("runtime", context);
    return make_runtime_hsaco_key(module, "gfx900", "+wavefrontsize64", true);
  };
  EXPECT_EQ(make_key(), make_key());
}

}  // namespace
}  // namespace quadrants::lang
