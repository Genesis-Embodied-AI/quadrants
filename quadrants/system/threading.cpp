/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "quadrants/system/threading.h"
#include "quadrants/system/thread_pool_wait.h"

#include <algorithm>
#include <condition_variable>
#include <thread>
#include <vector>

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

ThreadPool::ThreadPool(int max_num_threads, int worker_spin_us)
    : max_num_threads(max_num_threads), worker_spin_us_(worker_spin_us) {
  QD_ASSERT_INFO(worker_spin_us >= 0, "cpu_worker_spin_us must be >= 0");
  exiting = false;
  started = false;
  running_threads = 0;
  timestamp = 1;
  last_finished = 0;
  task_head = 0;
  task_tail = 0;
  desired_num_threads = 0;
  thread_counter = 0;
  threads.resize((std::size_t)max_num_threads);
  for (int i = 0; i < max_num_threads; i++) {
    threads[i] = std::thread([this] { this->target(); });
  }
}

void ThreadPool::run(int splits, int desired_num_threads, void *task_context, TaskFn *task_fn) {
  {
    std::lock_guard _(mutex);
    this->task_context = task_context;
    this->task_fn = task_fn;
    this->desired_num_threads = std::min(desired_num_threads, max_num_threads);
    QD_ASSERT(this->desired_num_threads > 0);
    // QD_P(this->desired_num_threads);
    started = false;
    if (greedy_.static_tasks)
      running_threads = this->desired_num_threads;
    if (greedy_.atomic_completion)
      active_atomic_.store(greedy_.static_tasks ? this->desired_num_threads : 0, std::memory_order_relaxed);
    task_head = 0;
    task_tail = splits;
    timestamp++;
    QD_ASSERT(timestamp < (1LL << 62));  // avoid overflowing here
    if (worker_spin_us_ > 0)
      wakeup_generation_.store(timestamp, std::memory_order_relaxed);
  }

  // wake up all slaves
  slave_cv.notify_all();
  if (greedy_.caller_poll) {
    // Completion publication is separate from completion counting, so caller polling can be tested on its own.
    while (completed_generation_.load(std::memory_order_acquire) != timestamp) {
    }
  } else {
    std::unique_lock<std::mutex> lock(mutex);
    master_cv.wait(lock, [this] {
      return greedy_.atomic_completion ? completed_generation_.load(std::memory_order_acquire) == timestamp
                                       : started && running_threads == 0;
    });
  }
  QD_ASSERT(greedy_.static_tasks || task_head >= task_tail);
}

void ThreadPool::target() {
  uint64 last_timestamp = 0;
  int thread_id;
  {
    std::lock_guard<std::mutex> lock(mutex);
    thread_id = thread_counter++;
  }
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      auto ready = [this, last_timestamp, thread_id] {
        return (timestamp > last_timestamp && thread_id < desired_num_threads) || this->exiting;
      };
      // Only workers that have observed a launch and are still eligible poll. New or excluded workers use the CV.
      if (worker_spin_us_ > 0 && last_timestamp != 0 && thread_id < desired_num_threads && !ready()) {
        auto observed = wakeup_generation_.load(std::memory_order_relaxed);
        lock.unlock();
        poll_thread_pool_generation(wakeup_generation_, observed, worker_spin_us_);
        lock.lock();
      }
      // Rechecking under the publisher's mutex prevents missed notifications at the polling-to-sleep boundary.
      slave_cv.wait(lock, ready);
      last_timestamp = timestamp;
      if (exiting) {
        break;
      } else {
        if (!greedy_.atomic_completion && last_finished >= last_timestamp) {
          continue;
          // This could happen when part of the desired threads wake up and
          // finish all the task, and then this thread wake up finding nothing
          // to do. Should skip this task directly.
        } else {
          if (greedy_.atomic_completion && !greedy_.static_tasks) {
            // Preserve production's late-worker skipping. Registration races atomically with final completion.
            int active = active_atomic_.load(std::memory_order_acquire);
            while (active != closed_completion &&
                   !active_atomic_.compare_exchange_weak(active, active + 1, std::memory_order_acq_rel)) {
            }
            if (active == closed_completion)
              continue;
          }
          started = true;
          if (!greedy_.atomic_completion && !greedy_.static_tasks)
            running_threads++;
        }
      }
    }

    if (greedy_.static_tasks) {
      for (int task = thread_id; task < task_tail; task += desired_num_threads)
        task_fn(this->task_context, thread_id, task);
    } else {
      while (true) {
        // For a single parallel task
        int task_id;
        {
          task_id = task_head.fetch_add(1, std::memory_order_relaxed);
          if (task_id >= task_tail)
            break;
        }

        task_fn(this->task_context, thread_id, task_id);
      }
    }

    bool all_finished = false;
    if (greedy_.atomic_completion) {
      int active = active_atomic_.load(std::memory_order_relaxed);
      while (!active_atomic_.compare_exchange_weak(active, active == 1 ? closed_completion : active - 1,
                                                   std::memory_order_acq_rel)) {
      }
      all_finished = active == 1;
      if (all_finished) {
        if (greedy_.caller_poll) {
          completed_generation_.store(last_timestamp, std::memory_order_release);
        } else {
          // Only the final worker locks. Pair CV publication with its predicate check to avoid missed wakeups.
          std::lock_guard<std::mutex> lock(mutex);
          completed_generation_.store(last_timestamp, std::memory_order_release);
        }
      }
    } else {
      std::lock_guard<std::mutex> lock(mutex);
      running_threads--;
      if (running_threads == 0) {
        all_finished = true;
        last_finished = last_timestamp;
        if (greedy_.caller_poll)
          completed_generation_.store(last_timestamp, std::memory_order_release);
      }
    }
    if (all_finished && !greedy_.caller_poll)
      master_cv.notify_one();
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lg(mutex);
    exiting = true;
    if (worker_spin_us_ > 0)
      wakeup_generation_.store(timestamp + 1, std::memory_order_relaxed);
  }
  slave_cv.notify_all();
  for (auto &th : threads)
    th.join();
}

}  // namespace quadrants
