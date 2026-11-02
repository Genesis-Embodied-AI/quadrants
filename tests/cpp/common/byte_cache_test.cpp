#include <atomic>
#include <future>
#include <vector>

#include "gtest/gtest.h"
#include "quadrants/common/byte_cache.h"

namespace quadrants::lang {
namespace {

TEST(ByteCache, ConcurrentLookupBuildsOnce) {
  // Concurrent requests for one key must build once and return the same bytes, including embedded zeros.
  ByteCache<std::string> cache;
  std::atomic<int> builds{0};
  const std::string binary("a\0b", 3);
  auto build = [&] {
    ++builds;
    return binary;
  };
  std::promise<void> start;
  auto ready = start.get_future().share();
  std::vector<std::future<std::string>> results;
  for (int i = 0; i < 8; ++i) {
    results.push_back(std::async(std::launch::async, [&] {
      ready.wait();
      bool hit;
      return cache.get_or_build("runtime", build, hit);
    }));
  }
  start.set_value();
  for (auto &result : results)
    EXPECT_EQ(result.get(), binary);
  EXPECT_EQ(builds, 1);
}

}  // namespace
}  // namespace quadrants::lang
