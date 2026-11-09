#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace quadrants {

// Experimental caller-participating pool for measuring issue #1080. Based on the launch protocol in Alexis Duburcq's
// duburcqa/cpu_multithreading_poc branch (d5428164516366ac60e5a0d23f7a96eeae831a50). Like ThreadPool, run() has one
// caller.
class LowLatencyThreadPool {
 public:
  using Task = void(void *, int, int);
  LowLatencyThreadPool(int threads, int spin_us, bool static_tasks);
  ~LowLatencyThreadPool();
  void run(int splits, int desired_threads, void *context, Task *task);

 private:
  void target(int id);
  void work(int id);
  int max_threads_;
  int spin_us_;
  bool static_tasks_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable cv_;
  alignas(64) std::atomic<std::uint64_t> launch_{0};
  alignas(64) std::atomic<int> next_task_{0};
  alignas(64) std::atomic<int> remaining_{0};
  alignas(64) std::atomic<int> sleeping_{0};
  std::atomic<bool> exiting_{false};
  int splits_{0};
  int active_threads_{0};
  void *context_{nullptr};
  Task *task_{nullptr};
};

}  // namespace quadrants
