#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace quadrants {

enum class ThreadPoolWaitMode { tight, pause, sleep1 };

// Benchmark-only selector; this branch is separate from the production polling PR.
inline ThreadPoolWaitMode benchmark_thread_pool_wait_mode() {
  const char *value = std::getenv("QD_BENCH_WORKER_WAIT");
  std::string_view name = value ? value : "pause";
  if (name == "tight")
    return ThreadPoolWaitMode::tight;
  if (name == "pause")
    return ThreadPoolWaitMode::pause;
  if (name == "sleep1")
    return ThreadPoolWaitMode::sleep1;
  throw std::invalid_argument("QD_BENCH_WORKER_WAIT must be tight, pause, or sleep1");
}

// This is only a wakeup hint. The caller must recheck its work/shutdown predicate under the pool mutex before
// accessing task data or blocking on the condition variable.
inline void poll_thread_pool_generation(const std::atomic<std::uint64_t> &generation,
                                        std::uint64_t observed,
                                        int spin_us,
                                        ThreadPoolWaitMode mode = ThreadPoolWaitMode::pause) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(spin_us);
  unsigned polls = 0;
  while (generation.load(std::memory_order_relaxed) == observed) {
    if (mode == ThreadPoolWaitMode::sleep1) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    } else if (mode == ThreadPoolWaitMode::pause) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
      _mm_pause();
#elif defined(_MSC_VER) && defined(_M_ARM64)
      __yield();
#elif defined(__x86_64__) || defined(__i386__)
      __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
      asm volatile("yield" ::: "memory");
#endif
    }
    // Amortize clock reads. This is an approximate budget, not a scheduling deadline.
    if ((mode == ThreadPoolWaitMode::sleep1 || ++polls % 64 == 0) && std::chrono::steady_clock::now() >= deadline)
      return;
  }
}

}  // namespace quadrants
