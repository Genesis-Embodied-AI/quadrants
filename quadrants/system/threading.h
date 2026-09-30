/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "quadrants/common/core.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <thread>

namespace quadrants {

using RangeForTaskFunc = void(void *ctx, int thread_id, int task_id);
using ParallelFor = void(int n, int num_threads, void *, RangeForTaskFunc task_fn);

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
  RangeForTaskFunc *task_fn;
  void *range_for_task_context;  // Note: this is a pointer to a
                                 // range_task_helper_context defined in the
                                 // LLVM runtime, which is different from
                                 // quadrants::lang::Context.
  int thread_counter;

  explicit ThreadPool(int max_num_threads);

  void run(int splits, int desired_num_threads, void *range_for_task_context, RangeForTaskFunc *task_fn);

  static void static_run(ThreadPool *pool,
                         int splits,
                         int desired_num_threads,
                         void *range_for_task_context,
                         RangeForTaskFunc *task_fn) {
    return pool->run(splits, desired_num_threads, range_for_task_context, task_fn);
  }

  void target();

  ~ThreadPool();
};

}  // namespace quadrants
