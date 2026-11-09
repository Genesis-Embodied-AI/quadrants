#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <thread>

#include "quadrants/system/controlled_thread_pool.h"

namespace quadrants {

TEST(ControlledThreadPool, MechanismCombinationsAndSleepTransitions) {
  using Pool = ControlledThreadPool;
  using Wait = Pool::Wait;
  for (bool caller_work : {false, true}) {
    for (bool atomic_completion : {false, true}) {
      for (bool fast_launch : {false, true}) {
        for (bool static_tasks : {false, true}) {
          for (Wait caller_wait : {Wait::cv, Wait::tight, Wait::pause, Wait::sleep1}) {
            if (!atomic_completion && caller_wait != Wait::cv)
              continue;
            for (Wait worker_wait : {Wait::cv, Wait::tight, Wait::pause, Wait::sleep1}) {
              for (int spin_us : {0, 20, 200, -1}) {
                Pool::Options options;
                options.caller_work = caller_work;
                options.atomic_completion = atomic_completion;
                options.fast_launch = fast_launch;
                options.static_tasks = static_tasks;
                options.caller_wait = caller_wait;
                options.worker_wait = worker_wait;
                options.spin_us = spin_us;
                options.inline_single = spin_us == 20 || spin_us == -1;
                Pool pool(4, options);
                struct Context {
                  std::array<std::atomic<int>, 127> counts;
                  std::array<int, 127> values;
                  int generation;
                  int active_threads;
                  bool static_tasks;
                  bool caller_work;
                  std::thread::id caller;
                  std::atomic<bool> bad{false};
                } context;
                context.static_tasks = static_tasks;
                context.caller_work = caller_work;
                context.caller = std::this_thread::get_id();
                constexpr std::array<int, 6> sizes{0, 1, 2, 7, 32, 127};
                for (int generation = 1; generation <= 48; ++generation) {
                  int size = sizes[generation % sizes.size()];
                  int threads = 1 << (generation % 4);
                  context.generation = generation;
                  context.active_threads = std::min({size, threads, 4});
                  context.values.fill(-1);
                  for (auto &count : context.counts)
                    count.store(0);
                  if (generation % 12 == 0)
                    std::this_thread::sleep_for(std::chrono::microseconds(400));
                  pool.run(size, threads, &context, [](void *raw, int thread, int task) {
                    auto &ctx = *static_cast<Context *>(raw);
                    if (thread < 0 || thread >= ctx.active_threads ||
                        (ctx.static_tasks && thread != task % ctx.active_threads) ||
                        ((std::this_thread::get_id() == ctx.caller) != (ctx.caller_work && thread == 0))) {
                      ctx.bad.store(true);
                    }
                    if (task % 17 == 0)
                      std::this_thread::yield();
                    ctx.counts[task].fetch_add(1);
                    ctx.values[task] = ctx.generation * 1000 + task;
                  });
                  ASSERT_FALSE(context.bad.load());
                  for (int task = 0; task < 127; ++task) {
                    ASSERT_EQ(context.counts[task].load(), task < size ? 1 : 0);
                    ASSERT_EQ(context.values[task], task < size ? generation * 1000 + task : -1);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace quadrants
