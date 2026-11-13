#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace quadrants {

// This is only a wakeup hint. The caller must recheck its work/shutdown predicate under the pool mutex before
// accessing task data or blocking on the condition variable.
inline void poll_thread_pool_generation(const std::atomic<std::uint64_t> &generation,
                                        std::uint64_t observed,
                                        int spin_us) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(spin_us);
  unsigned polls = 0;
  while (generation.load(std::memory_order_relaxed) == observed) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(_MSC_VER) && defined(_M_ARM64)
    __yield();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#endif
    // Amortize clock reads. This is an approximate budget, not a scheduling deadline.
    if (++polls % 64 == 0 && std::chrono::steady_clock::now() >= deadline)
      return;
  }
}

}  // namespace quadrants
