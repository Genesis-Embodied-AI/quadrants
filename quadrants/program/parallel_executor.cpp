#include <quadrants/system/timeline.h>
#include "quadrants/program/parallel_executor.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <atomic>

namespace quadrants::lang {

static bool pexec_log() {
  static const bool v = []() {
    const char *e = std::getenv("QD_TPLOG");
    return e != nullptr && std::string(e) == "1";
  }();
  return v;
}
static std::atomic<int> g_pexec_task_seq{0};

ParallelExecutor::ParallelExecutor(const std::string &name, int num_threads)
    : name_(name), num_threads_(num_threads), status_(ExecutorStatus::uninitialized), running_threads_(0) {
  if (num_threads <= 0) {
    return;
  }
  {
    auto _ = std::lock_guard<std::mutex>(mut_);

    for (int i = 0; i < num_threads; i++) {
      threads_.emplace_back([this]() { this->worker_loop(); });
    }

    status_ = ExecutorStatus::initialized;
  }
  init_cv_.notify_all();
}

ParallelExecutor::~ParallelExecutor() {
  // TODO: We should have a new ExecutorStatus, e.g. shutting_down, to prevent
  // new tasks from being enqueued during shut down.
  if (num_threads_ <= 0) {
    return;
  }
  flush();
  {
    auto _ = std::lock_guard<std::mutex>(mut_);
    status_ = ExecutorStatus::finalized;
  }
  // Signal the workers that they need to shutdown.
  worker_cv_.notify_all();
  for (auto &th : threads_) {
    th.join();
  }
}

void ParallelExecutor::enqueue(const TaskType &func) {
  if (num_threads_ <= 0) {
    func();
    return;
  }
  {
    std::lock_guard<std::mutex> _(mut_);
    task_queue_.push_back(func);
  }
  worker_cv_.notify_all();
}

void ParallelExecutor::flush() {
  if (num_threads_ <= 0) {
    return;
  }
  std::unique_lock<std::mutex> lock(mut_);
  if (pexec_log()) {
    std::printf("[PEXEC:%s] flush ENTER queued=%zu running=%d\n", name_.c_str(), task_queue_.size(), running_threads_);
    std::fflush(stdout);
  }
  while (!flush_cv_cond()) {
    flush_cv_.wait(lock);
  }
  if (pexec_log()) {
    std::printf("[PEXEC:%s] flush EXIT\n", name_.c_str());
    std::fflush(stdout);
  }
}

bool ParallelExecutor::flush_cv_cond() {
  return (task_queue_.empty() && running_threads_ == 0);
}

void ParallelExecutor::worker_loop() {
  QD_DEBUG("Starting worker thread.");
  auto thread_id = thread_counter_++;

  std::string thread_name = name_;
  if (num_threads_ != 1)
    thread_name += fmt::format("_{}", thread_id);
  Timeline::get_this_thread_instance().set_name(thread_name);

  {
    std::unique_lock<std::mutex> lock(mut_);
    while (status_ == ExecutorStatus::uninitialized) {
      init_cv_.wait(lock);
    }
  }

  QD_DEBUG("Worker thread initialized and running.");
  bool done = false;
  while (!done) {
    bool notify_flush_cv = false;
    {
      std::unique_lock<std::mutex> lock(mut_);
      while (task_queue_.empty() && status_ == ExecutorStatus::initialized) {
        worker_cv_.wait(lock);
      }
      // So long as |task_queue| is not empty, we keep running.
      if (!task_queue_.empty()) {
        auto task = task_queue_.front();
        running_threads_++;
        task_queue_.pop_front();
        const bool log = pexec_log();
        const int seq = log ? ++g_pexec_task_seq : 0;
        lock.unlock();

        if (log) {
          std::printf("[PEXEC:%s] task #%d START\n", name_.c_str(), seq);
          std::fflush(stdout);
        }
        // Run the task
        task();
        if (log) {
          std::printf("[PEXEC:%s] task #%d END\n", name_.c_str(), seq);
          std::fflush(stdout);
        }

        lock.lock();
        running_threads_--;
      }
      notify_flush_cv = flush_cv_cond();
      if (status_ == ExecutorStatus::finalized && task_queue_.empty()) {
        done = true;
      }
    }
    if (notify_flush_cv) {
      // It is fine to notify |flush_cv_| while nobody is waiting on it.
      flush_cv_.notify_one();
    }
  }
}
}  // namespace quadrants::lang
