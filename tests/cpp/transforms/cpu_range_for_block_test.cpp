#include "gtest/gtest.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/program/compile_config.h"
#include "quadrants/runtime/llvm/runtime_module/cpu_range_for.h"

namespace quadrants::lang {

TEST(CPURangeForBlock, BlockBoundaries) {
  // 200 iterations, four workers, minimum block size 1.
  // Each block contains 50 iterations.
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 1, 0), 0);
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 1, 1), 50);
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 1, 2), 100);
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 1, 3), 150);
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 1, 4), 200);

  // Minimum block size 512: the first block contains all iterations.
  // The remaining blocks are empty.
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 512, 0), 0);
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 512, 1), 200);
  EXPECT_EQ(get_cpu_block_start_index(0, 200, 4, 512, 4), 200);
}

TEST(CPURangeForBlock, BlockBoundaryEdgeCases) {
  for (int cpu_min_block_size : {1, 16, 512, 2048, 1 << 30, std::numeric_limits<int32>::max()}) {
    // 200 iterations on 12 workers need blocks of 17, unless the configured minimum is larger.
    const int64_t expected_width = std::max(17, cpu_min_block_size);
    for (int boundary_index = 0; boundary_index <= 12; ++boundary_index) {
      EXPECT_EQ(get_cpu_block_start_index(0, 200, 12, cpu_min_block_size, boundary_index),
                std::min<int64_t>(200, expected_width * boundary_index));
    }
  }
}

TEST(CPURangeForBlock, FullSignedRange) {
  const int32 begin = std::numeric_limits<int32>::min();
  const int32 end = std::numeric_limits<int32>::max();
  const int32 expected[] = {begin, -(1 << 30), 0, 1 << 30, end};
  for (int boundary_index = 0; boundary_index <= 4; ++boundary_index) {
    EXPECT_EQ(get_cpu_block_start_index(begin, end, 4, 1, boundary_index), expected[boundary_index]);
    EXPECT_EQ(get_cpu_block_start_index(end, begin, 4, 1, boundary_index), begin);
  }
}

namespace {

// Keeps the compiler statements alive while the test inspects the transformed loop.
struct TransformedCPURange {
  std::unique_ptr<Block> root;
  OffloadedStmt *offloaded;
};

TransformedCPURange transform_cpu_range(int begin, int end, int num_threads, int cpu_min_block_size) {
  CompileConfig config;
  config.cpu_max_num_threads = num_threads;
  config.cpu_min_block_size = cpu_min_block_size;
  auto root = std::make_unique<Block>();
  auto *offloaded =
      root->insert(std::make_unique<OffloadedStmt>(OffloadedStmt::TaskType::range_for, config.arch, nullptr))
          ->as<OffloadedStmt>();
  offloaded->const_begin = true;
  offloaded->const_end = true;
  offloaded->begin_value = begin;
  offloaded->end_value = end;
  irpass::make_cpu_multithreaded_range_for(root.get(), config);
  irpass::type_check(root.get(), config);
  return {std::move(root), offloaded};
}

}  // namespace

TEST(CPURangeForBlock, UsesConfiguredMinimum) {
  for (int cpu_min_block_size : {1, 512}) {
    SCOPED_TRACE(cpu_min_block_size);
    auto loop = transform_cpu_range(/*begin=*/0, /*end=*/200, /*num_threads=*/4, cpu_min_block_size);

    // Check the number of runtime tasks described by the transformed outer loop.
    auto *outer = loop.offloaded;
    ASSERT_TRUE(outer->const_begin && outer->const_end);
    ASSERT_GT(outer->block_dim, 0);
    const int iterations = outer->end_value - outer->begin_value;
    const int task_count = (iterations + outer->block_dim - 1) / outer->block_dim;
    EXPECT_EQ(task_count, 4);

    // Each task runs a serial inner loop over its original iterations.
    ASSERT_FALSE(outer->body->statements.empty());
    auto *inner = outer->body->statements.back()->cast<RangeForStmt>();
    ASSERT_NE(inner, nullptr);
    EXPECT_TRUE(inner->strictly_serialized);

    // Both boundary calculations must receive the configured minimum.
    for (auto *bound : {inner->begin, inner->end}) {
      auto *call = bound->cast<InternalFuncStmt>();
      ASSERT_NE(call, nullptr);
      EXPECT_EQ(call->func_name, "get_cpu_block_start_index");
      ASSERT_EQ(call->args.size(), 5);
      auto *minimum = call->args[3]->cast<ConstStmt>();
      ASSERT_NE(minimum, nullptr);
      EXPECT_EQ(minimum->val.val_int(), cpu_min_block_size);
    }
  }
}

}  // namespace quadrants::lang
