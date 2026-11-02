#include "gtest/gtest.h"
#include "quadrants/runtime/llvm/runtime_bitcode_inputs.h"
#include "quadrants/runtime/llvm/runtime_bitcode_serialization.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"

namespace quadrants::lang {
namespace {

TEST(RuntimeBitcodeCache, InputsInvalidateByTargetPathAndEveryContent) {
  // Changing the target, file paths, or any input bytes must change the cache key.
  RuntimeBitcodeInputs original{"gfx1100", {{"runtime.bc", "runtime"}, {"ocml.bc", std::string("lib\0data", 8)}}};
  auto changed = original;
  changed.target = "gfx900";
  EXPECT_NE(original.key(), changed.key());
  for (const auto &[path, bytes] : original.files) {
    changed = original;
    changed.files[path] += "x";
    EXPECT_NE(original.key(), changed.key());
  }
  changed = original;
  changed.files["renamed.bc"] = changed.files.at("runtime.bc");
  changed.files.erase("runtime.bc");
  EXPECT_NE(original.key(), changed.key());
  EXPECT_EQ(original.key(), RuntimeBitcodeInputs(original).key());
  EXPECT_THROW(RuntimeBitcodeInputs::read("gfx1100", {"/nonexistent-quadrants-runtime/cache-test.bc"}),
               std::runtime_error);
}

TEST(RuntimeBitcodeCache, PreservesPreparedIntrinsicAttributesAcrossSerialization) {
  // Serialization must preserve the preparation flags needed for identical HSACO cache keys.
  llvm::LLVMContext source, destination;
  llvm::Module module("runtime", source);
  auto *intrinsic =
      llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::sqrt, {llvm::Type::getFloatTy(source)});
  intrinsic->addFnAttr(llvm::Attribute::AlwaysInline);
  std::string module_serialized = serialize_runtime_bitcode(module);
  EXPECT_EQ(module.getNamedMetadata(kRuntimeInlineIntrinsics), nullptr);
  auto parsed = parse_runtime_bitcode(module_serialized, destination);
  ASSERT_TRUE(bool(parsed));
  EXPECT_TRUE(parsed.get()->getFunction("llvm.sqrt.f32")->hasFnAttribute(llvm::Attribute::AlwaysInline));
  EXPECT_EQ(parsed.get()->getNamedMetadata(kRuntimeInlineIntrinsics), nullptr);
  EXPECT_EQ(module_serialized, serialize_runtime_bitcode(*parsed.get()));
}

TEST(RuntimeBitcodeCache, ModulesOutliveSourceContextAndMutateIndependently) {
  // Cached bytes must survive the source context and produce independent mutable modules.
  RuntimeBitcodeCache cache;
  bool hit;
  auto bytes = cache.get_or_build(
      "runtime",
      [] {
        llvm::LLVMContext source;
        llvm::Module module("source", source);
        auto type = llvm::FunctionType::get(llvm::Type::getInt32Ty(source), false);
        auto function = llvm::Function::Create(type, llvm::Function::ExternalLinkage, "value", module);
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(source, "entry", function));
        builder.CreateRet(builder.getInt32(7));
        std::string module_serialized;
        llvm::raw_string_ostream module_serialized_writer(module_serialized);
        llvm::WriteBitcodeToFile(module, module_serialized_writer);
        return module_serialized;
      },
      hit);
  llvm::LLVMContext first_context, second_context;
  auto first = llvm::parseBitcodeFile(llvm::MemoryBufferRef(bytes, "first"), first_context);
  ASSERT_TRUE(bool(first));
  auto &first_ret = first.get()->getFunction("value")->getEntryBlock().back();
  first_ret.setOperand(0, llvm::ConstantInt::get(llvm::Type::getInt32Ty(first_context), 99));
  auto cached = cache.get_or_build("runtime", [] { return "wrong"; }, hit);
  ASSERT_TRUE(hit);
  auto second = llvm::parseBitcodeFile(llvm::MemoryBufferRef(cached, "second"), second_context);
  ASSERT_TRUE(bool(second));
  auto value = llvm::cast<llvm::ConstantInt>(second.get()->getFunction("value")->getEntryBlock().back().getOperand(0));
  EXPECT_EQ(value->getZExtValue(), 7);
  EXPECT_NE(&first.get()->getContext(), &second.get()->getContext());
}

}  // namespace
}  // namespace quadrants::lang
