#include "gtest/gtest.h"

#include <algorithm>

#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/program/compile_config.h"

namespace quadrants::lang {

// Inspect the generated chunk width, rather than relying on timing or on how many workers the OS wakes up.
TEST(CPURangeForBlock, ChunkWidth) {
  for (int minimum : {1, 16, 512, 2048}) {
    CompileConfig config;
    config.cpu_max_num_threads = 12;
    config.cpu_min_range_for_block = minimum;
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
    irpass::constant_fold(&root);

    auto *inner = offloaded->body->statements.back()->as<RangeForStmt>();
    auto *end_min = inner->end->as<BinaryOpStmt>();
    ASSERT_EQ(end_min->op_type, BinaryOpType::min);
    auto *end_add = end_min->lhs->cast<BinaryOpStmt>();
    if (!end_add) {
      end_add = end_min->rhs->cast<BinaryOpStmt>();
    }
    ASSERT_NE(end_add, nullptr);
    ASSERT_EQ(end_add->op_type, BinaryOpType::add);
    auto *width = end_add->rhs->as<ConstStmt>();
    EXPECT_EQ(width->val.val_int(), std::max(17, minimum));
    EXPECT_EQ(offloaded->end_value, 12);
    EXPECT_EQ(offloaded->block_dim, 1);
  }
}

}  // namespace quadrants::lang
