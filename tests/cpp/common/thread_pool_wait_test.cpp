#include "gtest/gtest.h"
#include "quadrants/system/threading.h"
#include "quadrants/system/thread_pool_wait.h"

#include <array>
#include <chrono>
#include <thread>

namespace quadrants {
namespace {
struct Work {
  std::array<std::atomic<int>, 137> visits{};
  std::array<int, 137> values{};
  std::atomic<bool> invalid_thread{false};
  int threads;
  int generation;
};

void record_work(void *context, int thread, int task) {
  auto &work = *static_cast<Work *>(context);
  if (thread < 0 || thread >= work.threads)
    work.invalid_thread.store(true);
  work.visits[task].fetch_add(1);
  work.values[task] = work.generation * 1000 + task;
}
}  // namespace

TEST(ThreadPoolWait, RepeatedLaunchesAndSleepTransitions) {
  for (int spin_us : {0, 20, 200, 1000}) {
    ThreadPool pool(8, spin_us);
    Work work;
    for (int generation = 1; generation <= 500; ++generation) {
      work.threads = std::array<int, 4>{1, 8, 2, 4}[generation % 4];
      work.generation = generation;
      int tasks = std::array<int, 5>{0, 1, 3, 17, 137}[generation % 5];
      for (int i = 0; i < 137; ++i) {
        work.visits[i] = 0;
        work.values[i] = -1;
      }
      // Mix back-to-back launches with gaps around the polling deadline and long enough for CV sleeping.
      if (generation % 17 == 0)
        std::this_thread::sleep_for(std::chrono::microseconds(spin_us));
      if (generation % 29 == 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      pool.run(tasks, work.threads, &work, record_work);
      ASSERT_FALSE(work.invalid_thread.load());
      for (int i = 0; i < 137; ++i) {
        ASSERT_EQ(work.visits[i].load(), int(i < tasks));
        ASSERT_EQ(work.values[i], i < tasks ? generation * 1000 + i : -1);
      }
    }
  }
}

TEST(ThreadPoolWait, PublicationInterruptsLongPollingBudget) {
  std::atomic<std::uint64_t> generation{0};
  std::atomic<bool> entered{false};
  std::thread worker([&] {
    entered.store(true);
    poll_thread_pool_generation(generation, 0, 10000000);
  });
  while (!entered.load())
    std::this_thread::yield();
  auto start = std::chrono::steady_clock::now();
  generation.store(1);
  worker.join();
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(2));
}

TEST(ThreadPoolWait, PollingBudgetExpiresWithoutPublication) {
  std::atomic<std::uint64_t> generation{0};
  auto start = std::chrono::steady_clock::now();
  poll_thread_pool_generation(generation, 0, 200);
  auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_GE(elapsed, std::chrono::microseconds(200));
  EXPECT_LT(elapsed, std::chrono::seconds(2));
}

TEST(ThreadPoolWait, ShutdownAfterActiveAndIdleLaunches) {
  for (bool idle : {false, true}) {
    auto start = std::chrono::steady_clock::now();
    {
      ThreadPool pool(4, 10000000);
      Work work;
      work.threads = 4;
      work.generation = 1;
      pool.run(137, 4, &work, record_work);
      if (idle)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(2));
  }
  // No launch: newly constructed workers must also shut down cleanly.
  ThreadPool never_used(4, 200);
}

}  // namespace quadrants
