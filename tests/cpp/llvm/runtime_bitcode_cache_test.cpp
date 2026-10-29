#include "gtest/gtest.h"
#include "quadrants/runtime/llvm/runtime_bitcode_cache.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <thread>

namespace quadrants::lang {
namespace {

TEST(RuntimeBitcodeCache, InputsInvalidateByTargetPathAndEveryContent) {
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

TEST(RuntimeBitcodeCache, BinaryOwnershipAndConcurrentBuild) {
  RuntimeBitcodeCache cache;
  std::atomic<int> builds{0};
  const std::string binary("abc\0def", 7);
  std::vector<std::thread> threads;
  std::vector<std::string> results(8);
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&, i] {
      bool hit;
      results[i] = cache.get_or_build(
          "key",
          [&] {
            ++builds;
            return binary;
          },
          hit);
    });
  }
  for (auto &thread : threads)
    thread.join();
  EXPECT_EQ(builds, 1);
  for (const auto &result : results)
    EXPECT_EQ(result, binary);
  results[0][0] = 'z';
  bool hit;
  EXPECT_EQ(cache.get_or_build("key", [] { return "wrong"; }, hit), binary);
  EXPECT_TRUE(hit);
}

TEST(RuntimeBitcodeCache, FailureAndEmptyResultsAreRetried) {
  RuntimeBitcodeCache cache;
  bool hit;
  EXPECT_THROW(cache.get_or_build(
                   "key", []() -> std::string { throw std::runtime_error("build failed"); }, hit),
               std::runtime_error);
  EXPECT_EQ(cache.get_or_build("key", [] { return std::string(); }, hit), "");
  EXPECT_FALSE(hit);
  EXPECT_EQ(cache.get_or_build("key", [] { return "success"; }, hit), "success");
  EXPECT_FALSE(hit);
  cache.get_or_build("key", [] { return "wrong"; }, hit);
  EXPECT_TRUE(hit);
}

TEST(RuntimeBitcodeCache, EntryAndPayloadLimitsAndOversizedBypass) {
  bool hit;
  for (auto limits : {std::pair<std::size_t, std::size_t>{1, 100}, {4, 3}}) {
    RuntimeBitcodeCache cache(limits.first, limits.second);
    auto retained = cache.get_or_build("a", [] { return "aa"; }, hit);
    cache.get_or_build("b", [] { return "bb"; }, hit);
    EXPECT_EQ(cache.get_or_build("a", [] { return "new"; }, hit), "new");
    EXPECT_FALSE(hit);
    EXPECT_EQ(retained, "aa");
  }
  RuntimeBitcodeCache cache(4, 3);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(cache.get_or_build("large", [] { return "oversized"; }, hit), "oversized");
    EXPECT_FALSE(hit);
  }
}

TEST(RuntimeBitcodeCache, ModulesOutliveSourceContextAndMutateIndependently) {
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
        std::string result;
        llvm::raw_string_ostream stream(result);
        llvm::WriteBitcodeToFile(module, stream);
        return result;
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
