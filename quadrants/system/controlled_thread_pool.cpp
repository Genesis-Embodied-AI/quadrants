#include "quadrants/system/controlled_thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace quadrants {
namespace {
using Wait = ControlledThreadPool::Wait;
constexpr int worker_bits = 16;
constexpr std::uint64_t worker_mask = (std::uint64_t(1) << worker_bits) - 1;

void poll_wait(Wait kind) {
  if (kind == Wait::sleep1) {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  } else if (kind == Wait::pause) {
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
}

bool flag(const char *name) {
  const char *raw = std::getenv(name);
  if (!raw || std::string(raw) == "0") {
    return false;
  }
  if (std::string(raw) != "1") {
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
  }
  return true;
}

Wait wait_option(const char *name) {
  const char *raw = std::getenv(name);
  std::string value = raw ? raw : "cv";
  if (value == "cv")
    return Wait::cv;
  if (value == "tight")
    return Wait::tight;
  if (value == "pause")
    return Wait::pause;
  if (value == "sleep1")
    return Wait::sleep1;
  throw std::invalid_argument(std::string(name) + " must be cv, tight, pause, or sleep1");
}
}  // namespace

ControlledThreadPool::Options ControlledThreadPool::environment_options() {
  Options options;
  options.caller_work = flag("QD_ABLATION_CALLER_WORK");
  options.atomic_completion = flag("QD_ABLATION_ATOMIC_COMPLETION");
  options.fast_launch = flag("QD_ABLATION_FAST_LAUNCH");
  options.static_tasks = flag("QD_ABLATION_STATIC_TASKS");
  options.inline_single = flag("QD_ABLATION_INLINE_SINGLE");
  options.worker_wait = wait_option("QD_ABLATION_WORKER_WAIT");
  options.caller_wait = wait_option("QD_ABLATION_CALLER_WAIT");
  const char *spin = std::getenv("QD_ABLATION_SPIN_US");
  options.spin_us = spin ? std::stoi(spin) : 0;
  return options;
}

ControlledThreadPool::ControlledThreadPool(int threads, Options options) : max_threads_(threads), options_(options) {
  if (threads < 1 || threads > int(worker_mask) || options.spin_us < -1 ||
      (!options.atomic_completion && options.caller_wait != Wait::cv)) {
    throw std::invalid_argument("Invalid controlled pool configuration");
  }
  for (int id = options.caller_work ? 1 : 0; id < threads; ++id) {
    workers_.emplace_back([this, id] { target(id); });
  }
}

void ControlledThreadPool::work(int id) {
  if (options_.static_tasks) {
    for (int task = id; task < splits_; task += active_threads_) {
      task_(context_, id, task);
    }
  } else {
    for (;;) {
      int task = next_task_.fetch_add(1, std::memory_order_relaxed);
      if (task >= splits_)
        break;
      task_(context_, id, task);
    }
  }
}

void ControlledThreadPool::complete() {
  if (options_.atomic_completion) {
    if (remaining_.fetch_sub(1, std::memory_order_release) == 1 && options_.caller_wait == Wait::cv) {
      // Pair the final completion with the caller's predicate check to prevent a missed notification.
      std::lock_guard<std::mutex> lock(completion_mutex_);
      completion_cv_.notify_one();
    }
  } else {
    std::lock_guard<std::mutex> lock(completion_mutex_);
    if (--remaining_locked_ == 0)
      completion_cv_.notify_one();
  }
}

void ControlledThreadPool::run(int splits, int desired_threads, void *context, Task *task) {
  if (splits <= 0)
    return;
  active_threads_ = std::max(1, std::min({splits, desired_threads, max_threads_}));
  context_ = context;
  task_ = task;
  splits_ = splits;
  next_task_.store(0, std::memory_order_relaxed);
  if (options_.caller_work && options_.inline_single && active_threads_ == 1) {
    work(0);
    return;
  }
  int workers = active_threads_ - int(options_.caller_work);
  if (options_.atomic_completion) {
    remaining_.store(workers, std::memory_order_relaxed);
  } else {
    std::lock_guard<std::mutex> lock(completion_mutex_);
    remaining_locked_ = workers;
  }
  auto generation = (launch_.load(std::memory_order_relaxed) >> worker_bits) + 1;
  auto value = (generation << worker_bits) | std::uint64_t(active_threads_);
  if (options_.fast_launch) {
    // Sequential consistency ensures either the worker sees publication before blocking, or we see its sleep
    // registration and notify it. All configurations use the same generation and participation protocol.
    launch_.store(value);
    if (sleeping_.load() > 0) {
      std::lock_guard<std::mutex> lock(launch_mutex_);
      launch_cv_.notify_all();
    }
  } else {
    std::lock_guard<std::mutex> lock(launch_mutex_);
    launch_.store(value);
    launch_cv_.notify_all();
  }
  if (options_.caller_work)
    work(0);
  if (options_.caller_wait == Wait::cv) {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_cv_.wait(lock, [this] {
      return options_.atomic_completion ? remaining_.load(std::memory_order_acquire) == 0 : remaining_locked_ == 0;
    });
  } else {
    while (remaining_.load(std::memory_order_acquire) != 0)
      poll_wait(options_.caller_wait);
  }
}

void ControlledThreadPool::target(int id) {
  std::uint64_t last = 0;
  for (;;) {
    auto start = std::chrono::steady_clock::now();
    int spins = 0;
    auto launch = launch_.load(std::memory_order_acquire);
    while (launch == last && !exiting_.load(std::memory_order_relaxed)) {
      bool sleep = options_.worker_wait == Wait::cv || options_.spin_us == 0;
      if (!sleep) {
        poll_wait(options_.worker_wait);
        // Check after each requested sleep, or every 64 tight/PAUSE polls. A requested 1 us sleep can overshoot.
        if (options_.spin_us > 0 && (options_.worker_wait == Wait::sleep1 || ++spins % 64 == 0)) {
          sleep = std::chrono::steady_clock::now() - start >= std::chrono::microseconds(options_.spin_us);
        }
      }
      if (sleep) {
        std::unique_lock<std::mutex> lock(launch_mutex_);
        sleeping_.fetch_add(1);
        launch_cv_.wait(lock, [this, last] { return launch_.load() != last || exiting_.load(); });
        sleeping_.fetch_sub(1);
        start = std::chrono::steady_clock::now();
      }
      launch = launch_.load(std::memory_order_acquire);
    }
    if (exiting_.load(std::memory_order_relaxed))
      return;
    last = launch;
    if (id < int(launch & worker_mask)) {
      work(id);
      complete();
    }
  }
}

ControlledThreadPool::~ControlledThreadPool() {
  {
    std::lock_guard<std::mutex> lock(launch_mutex_);
    exiting_.store(true);
  }
  launch_cv_.notify_all();
  for (auto &worker : workers_)
    worker.join();
}
}  // namespace quadrants
