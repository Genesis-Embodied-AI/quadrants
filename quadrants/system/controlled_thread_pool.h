#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace quadrants {

// A common protocol with independently selectable mechanisms for CPU pool ablation experiments. Unlike the original
// pool's late-worker registration, every selected execution slot must finish each launch in all configurations here.
class ControlledThreadPool {
 public:
  enum class Wait { cv, tight, pause, sleep1 };
  struct Options {
    bool caller_work = false;
    bool atomic_completion = false;
    bool fast_launch = false;
    bool static_tasks = false;
    bool inline_single = false;
    Wait worker_wait = Wait::cv;
    Wait caller_wait = Wait::cv;
    int spin_us = 0;  // -1 means unbounded polling; zero means immediate condition-variable waiting.
  };
  using Task = void(void *, int, int);
  static Options environment_options();
  ControlledThreadPool(int threads, Options options);
  ~ControlledThreadPool();
  void run(int splits, int desired_threads, void *context, Task *task);

 private:
  void target(int id);
  void work(int id);
  void complete();
  int max_threads_;
  const Options options_;
  std::vector<std::thread> workers_;
  std::mutex launch_mutex_;
  std::condition_variable launch_cv_;
  std::mutex completion_mutex_;
  std::condition_variable completion_cv_;
  alignas(64) std::atomic<std::uint64_t> launch_{0};
  alignas(64) std::atomic<int> next_task_{0};
  alignas(64) std::atomic<int> remaining_{0};
  alignas(64) std::atomic<int> sleeping_{0};
  int remaining_locked_{0};
  std::atomic<bool> exiting_{false};
  int splits_{0};
  int active_threads_{0};
  void *context_{nullptr};
  Task *task_{nullptr};
};

}  // namespace quadrants
