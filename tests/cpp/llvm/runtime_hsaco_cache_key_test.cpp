#include "gtest/gtest.h"
#include "llvm/IR/Constants.h"
#include "quadrants/runtime/amdgpu/runtime_hsaco_cache_key.h"

namespace quadrants::lang {
namespace {

TEST(RuntimeHsacoCache, KeyTracksCompilationInputs) {
  llvm::LLVMContext context;
  llvm::Module module("runtime", context);
  auto value = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 1);
  auto global = new llvm::GlobalVariable(module, value->getType(), true, llvm::GlobalValue::ExternalLinkage, value);
  auto key = [&] { return make_runtime_hsaco_key(module, "gfx900", "+wavefrontsize64", true); };
  const auto original = key();
  EXPECT_EQ(original, key());
  EXPECT_FALSE(original == make_runtime_hsaco_key(module, "gfx1100", "+wavefrontsize64", true));
  EXPECT_FALSE(original == make_runtime_hsaco_key(module, "gfx900", "+wavefrontsize32", true));
  EXPECT_FALSE(original == make_runtime_hsaco_key(module, "gfx900", "+wavefrontsize64", false));
  global->setInitializer(llvm::ConstantInt::get(value->getType(), 2));
  EXPECT_FALSE(original == key());
  global->setInitializer(value);
  EXPECT_EQ(original, key());
}

}  // namespace
}  // namespace quadrants::lang
