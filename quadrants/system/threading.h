/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "quadrants/common/core.h"
#include "quadrants/system/thread_pool_greedy.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <thread>

namespace quadrants {

using TaskFn = void(void *ctx, int thread_id, int task_id);
using ParallelFor = void(int n, int num_threads, void *, TaskFn task_fn);

class ThreadPool {
 public:
  std::vector<std::thread> threads;
  std::condition_variable slave_cv;
  std::condition_variable master_cv;
  std::mutex mutex;
  std::atomic<int> task_head;
  int task_tail;
  int running_threads;
  int max_num_threads;
  int desired_num_threads;
  uint64 timestamp;
  uint64 last_finished;
  bool started;
  bool exiting;
  TaskFn *task_fn;
  // Opaque context passed to task_fn. LLVM runtime tasks use loop-specific helper contexts here, not
  // quadrants::lang::Context.
  void *task_context;
  int thread_counter;

  explicit ThreadPool(int max_num_threads, int worker_spin_us = greedy_default_worker_spin_us);

  void run(int splits, int desired_num_threads, void *task_context, TaskFn *task_fn);

  static void static_run(ThreadPool *pool, int splits, int desired_num_threads, void *task_context, TaskFn *task_fn) {
    return pool->run(splits, desired_num_threads, task_context, task_fn);
  }

  void target();

  ~ThreadPool();

 private:
  // A negative active count closes dynamic registration when the final participating worker finishes.
  static constexpr int closed_completion = -1;
  const ThreadPoolGreedyOptions greedy_;
  std::atomic<int> active_atomic_{0};
  std::atomic<uint64> completed_generation_{0};
  const int worker_spin_us_;
  std::atomic<std::uint64_t> wakeup_generation_{0};
};

}  // namespace quadrants
