#pragma once

#include <algorithm>
#include <cstdint>

namespace quadrants::lang {

inline int32_t get_cpu_block_start_index(int32_t range_begin,
                                         int32_t range_end,
                                         int32_t num_threads,
                                         int32_t min_block_size,
                                         int32_t boundary_index) {
  // Return the index at the start of the requested CPU block, clamped to range_end.
  // Block indices run from 0 to num_threads - 1. Boundary indices run from 0 to num_threads, because each block needs
  // a start and an end. The boundary at num_threads is range_end.

  // Widen before subtracting or multiplying: boundary can exceed i32, even if range_end cannot
  const int64_t count = std::max<int64_t>(0, int64_t(range_end) - range_begin);
  const int64_t block_size = std::max<int64_t>((count + num_threads - 1) / num_threads, min_block_size);
  const int64_t boundary = int64_t(range_begin) + block_size * boundary_index;
  // Unused blocks must be empty, including when their unclamped boundaries exceed i32.
  return static_cast<int32_t>(std::min<int64_t>(range_end, boundary));
}

}  // namespace quadrants::lang
