#include "quadrants/system/low_latency_thread_pool.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace quadrants {
namespace {
constexpr int worker_bits = 16;
constexpr std::uint64_t worker_mask = (std::uint64_t(1) << worker_bits) - 1;

void relax_cpu() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __yield();
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
  asm volatile("yield" ::: "memory");
#else
  std::this_thread::yield();
#endif
}
}  // namespace

LowLatencyThreadPool::LowLatencyThreadPool(int threads, int spin_us, bool static_tasks)
    : max_threads_(threads), spin_us_(spin_us), static_tasks_(static_tasks) {
  if (threads < 1 || threads > int(worker_mask) || spin_us < 0) {
    throw std::invalid_argument("Invalid CPU pool thread count or spin duration");
  }
  for (int id = 1; id < threads; ++id) {
    workers_.emplace_back([this, id] { target(id); });
  }
}

void LowLatencyThreadPool::work(int id) {
  if (static_tasks_) {
    for (int task = id; task < splits_; task += active_threads_) {
      task_(context_, id, task);
    }
  } else {
    for (;;) {
      int task = next_task_.fetch_add(1, std::memory_order_relaxed);
      if (task >= splits_) {
        break;
      }
      task_(context_, id, task);
    }
  }
}

void LowLatencyThreadPool::run(int splits, int desired_threads, void *context, Task *task) {
  if (splits <= 0) {
    return;
  }
  active_threads_ = std::max(1, std::min({splits, desired_threads, max_threads_}));
  context_ = context;
  task_ = task;
  splits_ = splits;
  next_task_.store(0, std::memory_order_relaxed);
  if (active_threads_ == 1) {
    work(0);
    return;
  }
  int workers = active_threads_ - 1;
  remaining_.store(workers, std::memory_order_relaxed);
  auto generation = (launch_.load(std::memory_order_relaxed) >> worker_bits) + 1;
  // Sequential consistency pairs publication with the sleeping counter and the worker's final predicate check.
  // Either a worker observes this generation before blocking, or we observe its sleep registration and notify it.
  launch_.store((generation << worker_bits) | std::uint64_t(workers));
  if (sleeping_.load() > 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    cv_.notify_all();
  }
  work(0);
  while (remaining_.load(std::memory_order_acquire) != 0) {
    relax_cpu();
  }
}

void LowLatencyThreadPool::target(int id) {
  std::uint64_t last = 0;
  for (;;) {
    auto start = std::chrono::steady_clock::now();
    int spins = 0;
    auto launch = launch_.load(std::memory_order_acquire);
    while (launch == last && !exiting_.load(std::memory_order_relaxed)) {
      relax_cpu();
      if (spin_us_ == 0 ||
          (++spins % 64 == 0 && std::chrono::steady_clock::now() - start >= std::chrono::microseconds(spin_us_))) {
        std::unique_lock<std::mutex> lock(mutex_);
        sleeping_.fetch_add(1);
        cv_.wait(lock, [this, last] { return launch_.load() != last || exiting_.load(); });
        sleeping_.fetch_sub(1);
        start = std::chrono::steady_clock::now();
      }
      launch = launch_.load(std::memory_order_acquire);
    }
    if (exiting_.load(std::memory_order_relaxed)) {
      return;
    }
    last = launch;
    if (id <= int(launch & worker_mask)) {
      work(id);
      remaining_.fetch_sub(1, std::memory_order_release);
    }
  }
}

LowLatencyThreadPool::~LowLatencyThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    exiting_.store(true);
  }
  cv_.notify_all();
  for (auto &worker : workers_) {
    worker.join();
  }
}
}  // namespace quadrants
