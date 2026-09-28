#include "gtest/gtest.h"

#include <algorithm>
#include <limits>

#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/program/compile_config.h"
#include "quadrants/runtime/llvm/runtime_module/cpu_range_for.h"

namespace quadrants::lang {

TEST(CPURangeForBlock, BlockBoundaries) {
  for (int cpu_min_block_size : {1, 16, 512, 2048, 1 << 30, std::numeric_limits<int32>::max()}) {
    // 200 iterations on 12 workers need blocks of 17, unless the configured minimum is larger.
    const int64_t expected_width = std::max(17, cpu_min_block_size);
    for (int boundary_index = 0; boundary_index <= 12; ++boundary_index) {
      EXPECT_EQ(cpu_range_for_boundary(0, 200, 12, cpu_min_block_size, boundary_index),
                std::min<int64_t>(200, expected_width * boundary_index));
    }
  }
}

TEST(CPURangeForBlock, FullSignedRange) {
  const int32 begin = std::numeric_limits<int32>::min();
  const int32 end = std::numeric_limits<int32>::max();
  const int32 expected[] = {begin, -(1 << 30), 0, 1 << 30, end};
  for (int boundary_index = 0; boundary_index <= 4; ++boundary_index) {
    EXPECT_EQ(cpu_range_for_boundary(begin, end, 4, 1, boundary_index), expected[boundary_index]);
    EXPECT_EQ(cpu_range_for_boundary(end, begin, 4, 1, boundary_index), begin);
  }
}

TEST(CPURangeForBlock, CallsBoundaryHelper) {
  for (int cpu_min_block_size : {1, 16, 512, 2048, 1 << 30, std::numeric_limits<int32>::max()}) {
    CompileConfig config;
    config.cpu_max_num_threads = 12;
    config.cpu_min_block_size = cpu_min_block_size;
    Block root;
    auto *offloaded =
        root.insert(std::make_unique<OffloadedStmt>(OffloadedStmt::TaskType::range_for, config.arch, nullptr))
            ->as<OffloadedStmt>();
    offloaded->const_begin = true;
    offloaded->const_end = true;
    offloaded->begin_value = 0;
    offloaded->end_value = 200;
    irpass::make_cpu_multithreaded_range_for(&root, config);
    irpass::type_check(&root, config);

    auto *inner = offloaded->body->statements.back()->as<RangeForStmt>();
    for (auto *bound : {inner->begin, inner->end}) {
      auto *call = bound->as<InternalFuncStmt>();
      EXPECT_EQ(call->func_name, "cpu_range_for_block_boundary");
      EXPECT_FALSE(call->with_runtime_context);
      EXPECT_EQ(call->ret_type, PrimitiveType::i32);
      ASSERT_EQ(call->args.size(), 5);
      EXPECT_EQ(call->args[0]->as<ConstStmt>()->val.val_int(), 0);
      EXPECT_EQ(call->args[1]->as<ConstStmt>()->val.val_int(), 200);
      EXPECT_EQ(call->args[2]->as<ConstStmt>()->val.val_int(), 12);
      EXPECT_EQ(call->args[3]->as<ConstStmt>()->val.val_int(), cpu_min_block_size);
    }
    auto *index = inner->begin->as<InternalFuncStmt>()->args[4]->as<LoopIndexStmt>();
    EXPECT_EQ(index->loop, offloaded);
    auto *next_block_index = inner->end->as<InternalFuncStmt>()->args[4]->as<BinaryOpStmt>();
    EXPECT_EQ(next_block_index->op_type, BinaryOpType::add);
    EXPECT_EQ(next_block_index->lhs, index);
    EXPECT_EQ(next_block_index->rhs->as<ConstStmt>()->val.val_int(), 1);
    EXPECT_TRUE(inner->strictly_serialized);
    EXPECT_EQ(offloaded->end_value, 12);
    EXPECT_EQ(offloaded->block_dim, 1);
  }
}

}  // namespace quadrants::lang
