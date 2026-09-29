#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "quadrants/runtime/llvm/runtime_module/cpu_range_for.h"

namespace quadrants::lang {

TEST(CPURangeForBlock, BlockBoundaries) {
  // 200 iterations, four workers, minimum block size 1.
  // Each block contains 50 iterations.
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 1, 0), 0);
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 1, 1), 50);
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 1, 2), 100);
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 1, 3), 150);
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 1, 4), 200);

  // Minimum block size 512: the first block contains all iterations.
  // The remaining blocks are empty.
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 512, 0), 0);
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 512, 1), 200);
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 512, 4), 200);
}

TEST(CPURangeForBlock, BlockBoundaryEdgeCases) {
  for (int cpu_min_block_size : {1, 16, 512, 2048, 1 << 30, std::numeric_limits<int32_t>::max()}) {
    // 200 iterations on 12 workers need blocks of 17, unless the configured minimum is larger.
    const int64_t expected_width = std::max(17, cpu_min_block_size);
    for (int boundary_index = 0; boundary_index <= 12; ++boundary_index) {
      EXPECT_EQ(get_cpu_block_start_index(0, 200, 12, cpu_min_block_size, boundary_index),
                std::min<int64_t>(200, expected_width * boundary_index));
    }
  }
}

TEST(CPURangeForBlock, FullSignedRange) {
  const int32_t begin = std::numeric_limits<int32_t>::min();
  const int32_t end = std::numeric_limits<int32_t>::max();
  const int32_t expected[] = {begin, -(1 << 30), 0, 1 << 30, end};
  for (int boundary_index = 0; boundary_index <= 4; ++boundary_index) {
    EXPECT_EQ(get_cpu_block_start_index(begin, end, 4, 1, boundary_index), expected[boundary_index]);
    EXPECT_EQ(get_cpu_block_start_index(end, begin, 4, 1, boundary_index), begin);
  }
}

}  // namespace quadrants::lang
