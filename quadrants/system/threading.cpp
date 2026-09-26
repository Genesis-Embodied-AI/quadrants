/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "quadrants/system/threading.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace quadrants {

bool test_threading() {
  auto tp = ThreadPool(20);
  for (int j = 0; j < 100; j++) {
    tp.run(10, j + 1, &j, [](void *j, int _thread_id, int i) {
      double ret = 0.0;
      for (int t = 0; t < 10000000; t++) {
        ret += t * 1e-20;
      }
      QD_P(int(i + ret + 10 * *(int *)j));
    });
  }
  return true;
}

namespace {

constexpr int kLaunchWorkersBits = 16;
constexpr uint64 kLaunchWorkersMask = (uint64(1) << kLaunchWorkersBits) - 1;
// How long an idle worker spins on the launch counter before it sleeps. It covers the host-side gap between two
// kernel launches, and bounds the time an idle pool keeps its cores busy.
constexpr auto kSpinDuration = std::chrono::microseconds(200);

inline void cpu_relax() {
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

ThreadPool::ThreadPool(int max_num_threads) : max_num_threads_(std::max(max_num_threads, 1)) {
  QD_ASSERT(max_num_threads_ <= int(kLaunchWorkersMask));
  workers_.reserve(max_num_threads_ - 1);
  for (int i = 0; i < max_num_threads_ - 1; i++) {
    workers_.emplace_back([this, i] { this->target(i); });
  }
}

void ThreadPool::work(int thread_id) {
  while (true) {
    int task_id = task_head_.fetch_add(1, std::memory_order_relaxed);
    if (task_id >= task_tail_)
      break;
    func_(range_for_task_context_, thread_id, task_id);
  }
}

void ThreadPool::run(int splits, int desired_num_threads, void *range_for_task_context, RangeForTaskFunc *func) {
  QD_ASSERT(desired_num_threads > 0);
  int n_threads = std::min({desired_num_threads, max_num_threads_, splits});
  range_for_task_context_ = range_for_task_context;
  func_ = func;
  task_tail_ = splits;
  task_head_.store(0, std::memory_order_relaxed);
  if (n_threads <= 1) {
    work(max_num_threads_ - 1);
    return;
  }

  int n_workers = n_threads - 1;
  n_running_workers_.store(n_workers, std::memory_order_relaxed);
  uint64 launch = launch_.load(std::memory_order_relaxed);
  launch_.store(((launch >> kLaunchWorkersBits) + 1) << kLaunchWorkersBits | uint64(n_workers));
  // A worker registers as sleeping under the mutex before it checks the launch counter a last time, so either it sees
  // the new launch or it is registered here and woken.
  if (n_sleeping_workers_.load() > 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    worker_cv_.notify_all();
  }

  work(max_num_threads_ - 1);
  while (n_running_workers_.load(std::memory_order_acquire) > 0) {
    cpu_relax();
  }
}

void ThreadPool::target(int thread_id) {
  uint64 last_launch = 0;
  while (true) {
    uint64 launch = launch_.load(std::memory_order_acquire);
    auto spin_start = std::chrono::steady_clock::now();
    int n_spins = 0;
    while (launch == last_launch && !exiting_.load(std::memory_order_relaxed)) {
      cpu_relax();
      // Reading the clock costs more than a pause, so it is only read every few spins
      if (++n_spins % 64 == 0 && std::chrono::steady_clock::now() - spin_start > kSpinDuration) {
        std::unique_lock<std::mutex> lock(mutex_);
        n_sleeping_workers_.fetch_add(1);
        worker_cv_.wait(lock, [this, last_launch] {
          return launch_.load() != last_launch || exiting_.load(std::memory_order_relaxed);
        });
        n_sleeping_workers_.fetch_sub(1);
        spin_start = std::chrono::steady_clock::now();
      }
      launch = launch_.load(std::memory_order_acquire);
    }
    if (exiting_.load(std::memory_order_relaxed))
      break;
    last_launch = launch;
    // The caller waits for every worker of a launch before it starts the next one, so a worker taking part cannot miss
    // its launch, and one left out of it has nothing to do
    if (thread_id < int(launch & kLaunchWorkersMask)) {
      work(thread_id);
      n_running_workers_.fetch_sub(1, std::memory_order_release);
    }
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    exiting_.store(true);
  }
  worker_cv_.notify_all();
  for (auto &worker : workers_)
    worker.join();
}

}  // namespace quadrants
