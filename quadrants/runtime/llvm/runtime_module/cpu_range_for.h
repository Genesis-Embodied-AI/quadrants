#pragma once

#include <algorithm>
#include <cstdint>

namespace quadrants::lang {

// Boundary indices run from 0 through num_threads. Adjacent boundaries delimit one worker's block.
inline int32_t cpu_range_for_boundary(int32_t begin,
                                      int32_t end,
                                      int32_t num_threads,
                                      int32_t cpu_min_block_size,
                                      int32_t boundary_index) {
  // Widen before subtracting or multiplying: valid i32 bounds can produce offsets that exceed i32.
  const int64_t count = std::max<int64_t>(0, int64_t(end) - begin);
  const int64_t block_size = std::max<int64_t>((count + num_threads - 1) / num_threads, cpu_min_block_size);
  const int64_t boundary = int64_t(begin) + block_size * boundary_index;
  // Unused blocks must be empty, including when their unclamped boundaries exceed i32.
  return static_cast<int32_t>(std::min<int64_t>(end, boundary));
}

}  // namespace quadrants::lang
