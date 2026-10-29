#include <atomic>
#include <future>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "quadrants/runtime/amdgpu/runtime_hsaco_cache.h"

namespace quadrants::lang {
namespace {
const RuntimeHsacoKey key{"runtime IR digest", "gfx900", "+wavefrontsize64,-wavefrontsize32", true};

TEST(RuntimeHsacoCache, CompilationInputsInvalidate) {
  RuntimeHsacoCache cache;
  bool hit;
  int compilations = 0;
  auto compile = [&] { return std::to_string(++compilations); };
  cache.get_or_compile(key, compile, hit);
  for (int field = 0; field < 4; ++field) {
    auto changed = key;
    if (field == 0)
      changed.ir_digest += "changed IR";
    if (field == 1)
      changed.mcpu = "gfx1100";
    if (field == 2)
      changed.features = "+wavefrontsize32";
    if (field == 3)
      changed.fast_math = false;
    auto bytes = cache.get_or_compile(changed, compile, hit);
    EXPECT_FALSE(hit);
    EXPECT_EQ(cache.get_or_compile(changed, compile, hit), bytes);
    EXPECT_TRUE(hit);
  }
  EXPECT_EQ(compilations, 5);
  EXPECT_EQ(cache.get_or_compile(key, compile, hit), "1");
  EXPECT_TRUE(hit);
}

TEST(RuntimeHsacoCache, ConcurrentLookupCompilesOnceAndReturnsOwnedBytes) {
  RuntimeHsacoCache cache;
  std::atomic<int> compilations{0};
  std::promise<void> start;
  auto ready = start.get_future().share();
  std::vector<std::future<std::string>> results;
  for (int i = 0; i < 8; ++i) {
    results.push_back(std::async(std::launch::async, [&] {
      ready.wait();
      bool hit;
      return cache.get_or_compile(
          key,
          [&] {
            ++compilations;
            return std::string("a\0b", 3);
          },
          hit);
    }));
  }
  start.set_value();
  auto first = results.front().get();
  first[0] = 'x';
  for (std::size_t i = 1; i < results.size(); ++i)
    EXPECT_EQ(results[i].get(), std::string("a\0b", 3));
  bool hit;
  EXPECT_EQ(cache.get_or_compile(key, [] { return std::string("wrong"); }, hit), std::string("a\0b", 3));
  EXPECT_TRUE(hit);
  EXPECT_EQ(compilations, 1);
}

TEST(RuntimeHsacoCache, FailedAndEmptyCompilationsAreRetried) {
  RuntimeHsacoCache cache;
  bool hit;
  EXPECT_THROW(cache.get_or_compile(
                   key, []() -> std::string { throw std::runtime_error("compile failed"); }, hit),
               std::runtime_error);
  EXPECT_EQ(cache.get_or_compile(key, [] { return std::string(); }, hit), "");
  EXPECT_FALSE(hit);
  EXPECT_EQ(cache.get_or_compile(key, [] { return std::string("retry"); }, hit), "retry");
  EXPECT_FALSE(hit);
  EXPECT_EQ(cache.get_or_compile(key, [] { return std::string("wrong"); }, hit), "retry");
  EXPECT_TRUE(hit);
}

TEST(RuntimeHsacoCache, EvictsOldestByEntryAndByteLimits) {
  for (const auto limits : {std::pair<std::size_t, std::size_t>{2, 100}, {10, 6}}) {
    RuntimeHsacoCache cache(limits.first, limits.second);
    bool hit;
    auto second = key;
    second.mcpu = "second";
    auto third = key;
    third.mcpu = "third";
    auto compile = [] { return std::string("abc"); };
    cache.get_or_compile(key, compile, hit);
    cache.get_or_compile(second, compile, hit);
    cache.get_or_compile(key, compile, hit);
    EXPECT_TRUE(hit);
    cache.get_or_compile(third, compile, hit);
    cache.get_or_compile(second, compile, hit);
    EXPECT_TRUE(hit);
    cache.get_or_compile(key, compile, hit);
    EXPECT_FALSE(hit);
  }
}

TEST(RuntimeHsacoCache, OversizedEntryDoesNotEvictExistingEntry) {
  RuntimeHsacoCache cache(2, 3);
  bool hit;
  cache.get_or_compile(key, [] { return std::string("abc"); }, hit);
  auto other = key;
  other.mcpu = "other";
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(cache.get_or_compile(other, [] { return std::string("large"); }, hit), "large");
    EXPECT_FALSE(hit);
  }
  EXPECT_EQ(cache.get_or_compile(key, [] { return std::string("bad"); }, hit), "abc");
  EXPECT_TRUE(hit);
}
}  // namespace
}  // namespace quadrants::lang
