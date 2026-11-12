#pragma once

#include <algorithm>
#include <cstdint>

namespace quadrants::lang {

inline int32_t get_cpu_per_worker_block_start_index(int32_t original_range_begin,
                                                    int32_t original_range_end,
                                                    int32_t num_blocks,
                                                    int32_t min_block_size,
                                                    int32_t boundary_index) {
  // Return the index at the start of the requested CPU block, clamped to original_range_end.
  // Block indices run from 0 to num_blocks - 1. Boundary indices run from 0 to num_blocks, because each block needs
  // a start and an end. The boundary at num_blocks is original_range_end.

  // Widen before subtracting or multiplying: boundary can exceed i32, even if original_range_end cannot
  const int64_t count = std::max<int64_t>(0, int64_t(original_range_end) - original_range_begin);
  const int64_t block_size = std::max<int64_t>((count + num_blocks - 1) / num_blocks, min_block_size);
  const int64_t boundary = int64_t(original_range_begin) + block_size * boundary_index;
  return static_cast<int32_t>(std::min<int64_t>(original_range_end, boundary));
}

}  // namespace quadrants::lang
