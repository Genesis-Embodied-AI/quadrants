#include "gtest/gtest.h"

#include <algorithm>

#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/program/compile_config.h"

namespace quadrants::lang {

// White-box check on `make_cpu_multithreaded_range_for`: verify the inner serial chunk width directly from the
// generated IR, rather than relying on timing or on how many worker threads the OS happens to wake up. The pass
// rewrites a range-for into an outer loop over `cpu_max_num_threads` whose body is an inner serial range-for of
// width `max(ceil((end - begin) / cpu_max_num_threads), cpu_min_range_for_block)`.
TEST(CpuMinRangeForBlock, ChunkWidthHonorsConfig) {
  constexpr int kThreads = 12;
  constexpr int kEnd = 200;
  constexpr int kCeil = (kEnd + kThreads - 1) / kThreads;  // ceil(200 / 12) == 17

  // Cover both branches of the max(): minimums below the ceil (17 dominates) and above it (the floor dominates).
  for (int minimum : {1, 16, 512, 2048}) {
    CompileConfig config;
    config.cpu_max_num_threads = kThreads;
    config.cpu_min_range_for_block = minimum;

    Block root;
    auto *offloaded =
        root.insert(std::make_unique<OffloadedStmt>(OffloadedStmt::TaskType::range_for, config.arch, nullptr))
            ->as<OffloadedStmt>();
    offloaded->const_begin = true;
    offloaded->const_end = true;
    offloaded->begin_value = 0;
    offloaded->end_value = kEnd;

    irpass::make_cpu_multithreaded_range_for(&root, config);
    irpass::type_check(&root, config);
    irpass::constant_fold(&root);

    // The outer offloaded loop now iterates over the thread count, with block_dim collapsed to 1.
    EXPECT_EQ(offloaded->end_value, kThreads);
    EXPECT_EQ(offloaded->block_dim, 1);

    // The inner serial loop ends at min(end, block_begin + width); pull `width` out of the `+` operand.
    auto *inner = offloaded->body->statements.back()->as<RangeForStmt>();
    auto *end_min = inner->end->as<BinaryOpStmt>();
    ASSERT_EQ(end_min->op_type, BinaryOpType::min);
    auto *block_end_add = end_min->lhs->cast<BinaryOpStmt>();
    if (block_end_add == nullptr) {
      block_end_add = end_min->rhs->cast<BinaryOpStmt>();
    }
    ASSERT_NE(block_end_add, nullptr);
    ASSERT_EQ(block_end_add->op_type, BinaryOpType::add);
    auto *width = block_end_add->rhs->as<ConstStmt>();
    EXPECT_EQ(width->val.val_int(), std::max(kCeil, minimum));
  }
}

}  // namespace quadrants::lang
