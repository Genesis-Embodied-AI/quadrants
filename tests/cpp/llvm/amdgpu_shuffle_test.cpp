#include "gtest/gtest.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "quadrants/runtime/llvm/amdgpu_shuffle.h"

namespace quadrants::lang {
namespace {

// Check the module-loader transformation without an AMD GPU. The two implementations have distinct callees so a
// stale forwarding stub or a caller left attached to the discarded implementation is visible in the resulting IR.
void check_shuffle_selection(AMDGPUShuffleMode mode) {
  llvm::LLVMContext context;
  llvm::Module module("shuffle", context);
  auto *i32 = llvm::Type::getInt32Ty(context);
  auto *type = llvm::FunctionType::get(i32, {i32, i32}, false);
  auto make_function = [&](const char *name) {
    return llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage, name, module);
  };
  auto *read = make_function("amdgpu_ds_bpermute");
  auto *swap = llvm::Function::Create(llvm::FunctionType::get(i32, {i32}, false), llvm::GlobalValue::ExternalLinkage,
                                      "amdgpu_permlane64", module);
  auto *entry = make_function("amdgpu_cross_half_shuffle_i32");
  auto *wave64 = make_function("amdgpu_shuffle_wave64_i32");
  auto *two_halves = make_function("amdgpu_shuffle_two_halves_i32");
  auto define_call = [&](llvm::Function *function, llvm::Function *callee) {
    llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
    auto *call = builder.CreateCall(callee, {function->getArg(0), function->getArg(1)});
    builder.CreateRet(call);
    return call;
  };
  define_call(wave64, read);
  llvm::IRBuilder<> half_builder(llvm::BasicBlock::Create(context, "entry", two_halves));
  half_builder.CreateRet(half_builder.CreateCall(swap, {two_halves->getArg(1)}));
  auto *ordinary = make_function("amdgpu_shuffle_i32");
  auto *ordinary_call = define_call(ordinary, entry);
  auto *up_call = define_call(make_function("amdgpu_shuffle_up_i32"), entry);
  auto *down_call = define_call(make_function("amdgpu_shuffle_down_i32"), entry);
  auto *type_call = define_call(make_function("amdgpu_shuffle_f32"), ordinary);

  auto *expected = mode == AMDGPUShuffleMode::Wave64 ? wave64 : two_halves;
  auto *selected = select_amdgpu_shuffle(module, mode);
  EXPECT_EQ(selected, expected);
  EXPECT_EQ(module.getFunction("amdgpu_cross_half_shuffle_i32"), expected);
  EXPECT_EQ(module.getFunction("amdgpu_shuffle_i32"), ordinary);
  EXPECT_EQ(ordinary_call->getCalledFunction(), expected);
  EXPECT_EQ(up_call->getCalledFunction(), expected);
  EXPECT_EQ(down_call->getCalledFunction(), expected);
  EXPECT_EQ(type_call->getCalledFunction(), ordinary);
  EXPECT_EQ(module.getFunction("amdgpu_shuffle_wave64_i32"), nullptr);
  EXPECT_EQ(module.getFunction("amdgpu_shuffle_two_halves_i32"), nullptr);
  EXPECT_EQ(module.getFunction("amdgpu_permlane64") != nullptr, mode == AMDGPUShuffleMode::TwoHalves);
  EXPECT_FALSE(llvm::verifyModule(module, &llvm::errs()));
}

TEST(AMDGPUShuffle, FullWaveSelectionRemovesTheHalfSwapAndRetargetsCallers) {
  check_shuffle_selection(AMDGPUShuffleMode::Wave64);
}

TEST(AMDGPUShuffle, TwoHalfSelectionKeepsTheSwapAndRetargetsCallers) {
  check_shuffle_selection(AMDGPUShuffleMode::TwoHalves);
}

}  // namespace
}  // namespace quadrants::lang
