#include <atomic>
#include <future>
#include <vector>

#include "gtest/gtest.h"
#include "quadrants/runtime/amdgpu/runtime_hsaco_cache.h"

namespace quadrants::lang {
namespace {

TEST(RuntimeHsacoCache, ConcurrentLookupCompilesOnce) {
  // Concurrent requests for one key must compile once and return the same bytes, including embedded zeros.
  RuntimeHsacoCache cache;
  std::atomic<int> compilations{0};
  const std::string binary("a\0b", 3);
  auto compile = [&] {
    ++compilations;
    return binary;
  };
  std::promise<void> start;
  auto ready = start.get_future().share();
  std::vector<std::future<std::string>> results;
  for (int i = 0; i < 8; ++i) {
    results.push_back(std::async(std::launch::async, [&] {
      ready.wait();
      bool hit;
      return cache.get_or_compile({"runtime", "gfx900", "", true}, compile, hit);
    }));
  }
  start.set_value();
  for (auto &result : results)
    EXPECT_EQ(result.get(), binary);
  EXPECT_EQ(compilations, 1);
}

TEST(RuntimeHsacoCache, RespectsEntryLimitRegardlessOfBinarySize) {
  // Evict the oldest entry at the count limit, but retain large binaries just like small ones.
  RuntimeHsacoCache cache(2);
  auto lookup = [&](const char *name, const std::string &bytes = "abc") {
    bool hit;
    EXPECT_EQ(cache.get_or_compile({name, "gfx900", "", true}, [&] { return bytes; }, hit), bytes);
    return hit;
  };
  EXPECT_FALSE(lookup("first"));
  EXPECT_FALSE(lookup("second"));
  EXPECT_TRUE(lookup("first"));
  EXPECT_FALSE(lookup("third"));
  EXPECT_TRUE(lookup("second"));
  EXPECT_FALSE(lookup("first"));
  const std::string large_binary(8 * 1024 * 1024 + 1, 'x');
  EXPECT_FALSE(lookup("large", large_binary));
  EXPECT_TRUE(lookup("large", large_binary));
  EXPECT_TRUE(lookup("first"));
}

}  // namespace
}  // namespace quadrants::lang
