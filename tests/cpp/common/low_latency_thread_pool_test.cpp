#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <thread>

#include "quadrants/system/low_latency_thread_pool.h"

namespace quadrants {

TEST(LowLatencyThreadPool, ConsecutiveLaunchesAndSleepTransitions) {
  for (bool static_tasks : {false, true}) {
    for (int spin_us : {0, 20, 200}) {
      LowLatencyThreadPool pool(8, spin_us, static_tasks);
      struct Context {
        std::array<std::atomic<int>, 127> counts;
        std::array<int, 127> values;
        int generation;
        std::atomic<bool> bad_id{false};
        int active_threads;
      } context;
      constexpr std::array<int, 6> sizes{0, 1, 2, 7, 32, 127};
      for (int generation = 1; generation <= 1000; ++generation) {
        int size = sizes[generation % sizes.size()];
        int threads = 1 << (generation % 4);
        context.generation = generation;
        context.active_threads = std::min(size, threads);
        context.values.fill(-1);
        for (auto &count : context.counts) {
          count.store(0);
        }
        if (generation % 17 == 0) {
          std::this_thread::sleep_for(std::chrono::microseconds(400));
        }
        pool.run(size, threads, &context, [](void *raw, int thread, int task) {
          auto &ctx = *static_cast<Context *>(raw);
          if (thread < 0 || thread >= ctx.active_threads) {
            ctx.bad_id.store(true);
          }
          ctx.counts[task].fetch_add(1);
          ctx.values[task] = ctx.generation * 1000 + task;
        });
        ASSERT_FALSE(context.bad_id.load());
        for (int task = 0; task < 127; ++task) {
          ASSERT_EQ(context.counts[task].load(), task < size ? 1 : 0);
          ASSERT_EQ(context.values[task], task < size ? generation * 1000 + task : -1);
        }
      }
    }
  }
}

TEST(LowLatencyThreadPool, SingleThreadRunsOnCallerAndStaticTasksKeepTheirWorker) {
  for (int max_threads : {1, 8}) {
    LowLatencyThreadPool pool(max_threads, 20, true);
    struct Context {
      std::array<std::thread::id, 8> owners;
    } context;
    auto caller = std::this_thread::get_id();
    pool.run(1, max_threads, &context, [](void *raw, int id, int task) {
      auto &ctx = *static_cast<Context *>(raw);
      EXPECT_EQ(id, 0);
      EXPECT_EQ(task, 0);
      ctx.owners[0] = std::this_thread::get_id();
    });
    EXPECT_EQ(context.owners[0], caller);
    std::array<std::thread::id, 8> first;
    for (int iteration = 0; iteration < 100; ++iteration) {
      pool.run(max_threads, max_threads, &context, [](void *raw, int id, int task) {
        auto &ctx = *static_cast<Context *>(raw);
        EXPECT_EQ(id, task);
        ctx.owners[task] = std::this_thread::get_id();
      });
      if (iteration == 0) {
        first = context.owners;
      }
      for (int task = 0; task < max_threads; ++task) {
        EXPECT_EQ(context.owners[task], first[task]);
      }
      EXPECT_EQ(context.owners[0], caller);
    }
  }
}

}  // namespace quadrants
