#pragma once

namespace quadrants::lang {

// Clear the result before entering the allocator: a device-side failure can stop execution before the assignment
// completes. Leaving the previous result intact would let the caller mistake an old allocation for the new one.
template <typename Result, typename Allocate>
void store_allocation_result(Result *result, Allocate allocate) {
  *result = 0;
  *result = allocate();
}

}  // namespace quadrants::lang
