#include <atomic>
#include <future>
#include <vector>

#include "gtest/gtest.h"
#include "quadrants/runtime/amdgpu/runtime_hsaco_cache.h"

namespace quadrants::lang {
namespace {

TEST(RuntimeHsacoCache, ConcurrentLookupCompilesOnce) {
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

TEST(RuntimeHsacoCache, RespectsEntryAndByteLimits) {
  for (const auto limits : {std::pair<std::size_t, std::size_t>{2, 100}, {10, 6}}) {
    RuntimeHsacoCache cache(limits.first, limits.second);
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
    EXPECT_FALSE(lookup("oversized", std::string(101, 'x')));
    EXPECT_FALSE(lookup("oversized", std::string(101, 'x')));
    EXPECT_TRUE(lookup("first"));
  }
}
}  // namespace
}  // namespace quadrants::lang
