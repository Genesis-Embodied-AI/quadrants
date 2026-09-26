/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "quadrants/common/core.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace quadrants {

using RangeForTaskFunc = void(void *, int thread_id, int i);
using ParallelFor = void(int n, int num_threads, void *, RangeForTaskFunc func);

// Runs the tasks of a CPU parallel loop on up to max_num_threads threads: the calling thread, with thread id
// max_num_threads - 1, and max_num_threads - 1 worker threads, with thread ids 0 to max_num_threads - 2. A worker
// waits for the next loop by spinning on the launch counter for a short while, then sleeps on a condition variable, so
// that back-to-back kernel launches do not pay a wake-up each.
class ThreadPool {
 public:
  explicit ThreadPool(int max_num_threads);

  void run(int splits, int desired_num_threads, void *range_for_task_context, RangeForTaskFunc *func);

  static void static_run(ThreadPool *pool,
                         int splits,
                         int desired_num_threads,
                         void *range_for_task_context,
                         RangeForTaskFunc *func) {
    return pool->run(splits, desired_num_threads, range_for_task_context, func);
  }

  ~ThreadPool();

 private:
  void target(int thread_id);
  void work(int thread_id);

  int max_num_threads_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable worker_cv_;
  // The launch counter in the high bits and the number of workers taking part in the launch in the low
  // kLaunchWorkersBits, published together so that a worker never pairs a launch with the worker count of another.
  std::atomic<uint64> launch_{0};
  std::atomic<int> task_head_{0};
  int task_tail_{0};
  std::atomic<int> n_running_workers_{0};
  std::atomic<int> n_sleeping_workers_{0};
  std::atomic<bool> exiting_{false};
  RangeForTaskFunc *func_{nullptr};
  // A pointer to a range_task_helper_context defined in the LLVM runtime, which is different from
  // quadrants::lang::Context.
  void *range_for_task_context_{nullptr};
};

}  // namespace quadrants
