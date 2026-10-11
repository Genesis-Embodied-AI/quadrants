#include "gtest/gtest.h"

#include <functional>
#include <stdexcept>
#include <type_traits>

#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"

namespace quadrants::lang {
namespace {

// Retaining a reference or iterator before a traversal must not expose an
// owning pointer that can be reset, moved out, or replaced behind the guard.
static_assert(std::is_same_v<decltype(std::declval<Block &>().statements), const stmt_vector &>);
static_assert(std::is_same_v<decltype(std::declval<Block &>()[0]), const pStmt &>);
static_assert(std::is_same_v<decltype(*std::declval<Block &>().find(nullptr)), const pStmt &>);
static_assert(std::is_same_v<decltype(*std::declval<Block &>().locate(0)), const pStmt &>);
static_assert(!std::is_move_constructible_v<Block>);
static_assert(!std::is_copy_constructible_v<Block::StatementListGuard>);
static_assert(!std::is_move_constructible_v<Block::StatementListGuard>);

pStmt constant(int value) {
  return Stmt::make_typed<ConstStmt>(TypedConstant(value));
}

TEST(StatementListGuard, RejectsMutationsBeforeChangingOwnershipOrUses) {
  Block block;
  auto *old = block.insert(constant(1));
  auto *other = block.insert(constant(2));
  auto *use = block.push_back<BinaryOpStmt>(BinaryOpType::add, old, old)->as<BinaryOpStmt>();
  const auto begin = block.statements.begin();
  const auto end = block.statements.end();
  auto replacement = constant(3);
  auto *replacement_ptr = replacement.get();
  VecStatement replacements;
  replacements.push_back(constant(4));
  auto *vec_ptr = replacements[0].get();
  const auto guard = block.lock_statements();

  auto rejected = [&](const std::function<void()> &mutation) {
    EXPECT_THROW(mutation(), std::logic_error);
    ASSERT_EQ(block.size(), 3);
    EXPECT_EQ(block[0].get(), old);
    EXPECT_EQ(block[1].get(), other);
    EXPECT_EQ(block[2].get(), use);
    EXPECT_EQ(block.statements.begin(), begin);
    EXPECT_EQ(block.statements.end(), end);
    EXPECT_FALSE(old->erased);
    EXPECT_TRUE(block.trash_bin.empty());
    EXPECT_EQ(use->lhs, old);
    EXPECT_EQ(use->rhs, old);
    EXPECT_EQ(replacement.get(), replacement_ptr);
    EXPECT_EQ(replacement->parent, nullptr);
    ASSERT_EQ(replacements.size(), 1);
    EXPECT_EQ(replacements[0].get(), vec_ptr);
    EXPECT_EQ(vec_ptr->parent, nullptr);
  };

  rejected([&] { block.insert(std::move(replacement)); });
  rejected([&] { block.insert_at(std::move(replacement), begin); });
  rejected([&] { block.insert(std::move(replacements)); });
  rejected([&] { block.insert_at(std::move(replacements), begin); });
  rejected([&] { block.push_back<ConstStmt>(TypedConstant(5)); });
  rejected([&] { block.erase(0); });
  rejected([&] { block.erase(old); });
  rejected([&] { block.erase_range(begin, end); });
  rejected([&] { block.erase(std::unordered_set<Stmt *>{old}); });
  rejected([&] { block.extract(0); });
  rejected([&] { block.extract(old); });
  rejected([&] { block.extract_statements(); });
  rejected([&] { block.set_statements(std::move(replacements)); });
  rejected([&] { block.replace_statements_in_range(0, 1, std::move(replacements)); });
  // A one-for-one replacement must fail even though neither size nor capacity changes.
  rejected([&] { block.replace_with(old, std::move(replacement)); });
  rejected([&] { block.replace_with(old, std::move(replacements)); });
  rejected([&] { block.insert_before(old, std::move(replacements)); });
  rejected([&] { block.insert_after(old, std::move(replacements)); });
}

TEST(StatementListGuard, AllowsOperandUpdatesAndReleasesAfterNestedException) {
  Block block;
  auto *old = block.insert(constant(1));
  auto *replacement = block.insert(constant(2));
  auto *use = block.push_back<BinaryOpStmt>(BinaryOpType::add, old, old)->as<BinaryOpStmt>();
  {
    const auto outer = block.lock_statements();
    EXPECT_THROW(
        {
          const auto inner = block.lock_statements();
          block.erase(0);
        },
        std::logic_error);
    EXPECT_THROW(block.erase(0), std::logic_error);
    EXPECT_TRUE(use->replace_operand_with(old, replacement));
    EXPECT_EQ(use->lhs, replacement);
    EXPECT_EQ(use->rhs, replacement);
    EXPECT_FALSE(use->replace_operand_with(old, replacement));
    // Replacing an operand by itself still reports a match, as CSE expects.
    EXPECT_TRUE(use->replace_operand_with(replacement, replacement));
  }
  EXPECT_NO_THROW(block.erase(old));
  EXPECT_EQ(block.size(), 2);
}

TEST(StatementListGuard, BulkTransferAndCloneHaveIndependentStorageAndGuards) {
  Block source;
  auto *stmt = source.insert(constant(1));
  Block target;
  auto moved = source.extract_statements();
  EXPECT_TRUE(source.statements.empty());
  target.insert(VecStatement(std::move(moved)));
  EXPECT_EQ(target[0].get(), stmt);
  EXPECT_EQ(stmt->parent, &target);
  const auto guard = target.lock_statements();
  auto clone = target.clone();
  EXPECT_NO_THROW(clone->erase(0));
  EXPECT_EQ(target.size(), 1);
  EXPECT_THROW(target.extract_statements(), std::logic_error);
}

class CallbackStmt : public ConstStmt {
 public:
  explicit CallbackStmt(std::function<void()> action) : ConstStmt(TypedConstant(7)), action_(std::move(action)) {
  }

  void accept(IRVisitor *visitor) override {
    action_();
    ConstStmt::accept(visitor);
  }

 private:
  std::function<void()> action_;
};

TEST(StatementListGuard, UsageReplacementKeepsLeafDispatchAndGuardsAncestors) {
  Block root;
  auto *old = root.insert(constant(1));
  auto *replacement = root.insert(constant(2));
  auto *branch = root.push_back<IfStmt>(old)->as<IfStmt>();
  auto child = std::make_unique<Block>();
  child->insert(std::make_unique<CallbackStmt>([&] { root.erase(old); }));
  branch->set_true_statements(std::move(child));
  EXPECT_THROW(irpass::replace_all_usages_with(&root, old, replacement), std::logic_error);
  EXPECT_EQ(root.size(), 3);
  EXPECT_EQ(root[0].get(), old);
  // A rejected callback must not leave the traversal guard stuck on either block.
  EXPECT_NO_THROW(branch->true_statements->erase(0));
  EXPECT_NO_THROW(root.erase(old));
}

TEST(StatementListGuard, CseReplacementRejectsMutationDuringItsNoCopyWalk) {
  Block root;
  root.insert(constant(1));
  root.insert(constant(1));
  root.insert(std::make_unique<CallbackStmt>([&] { root.insert(constant(9)); }));
  EXPECT_THROW(irpass::whole_kernel_cse(&root), std::logic_error);
  EXPECT_EQ(root.size(), 3);
  EXPECT_NO_THROW(root.insert(constant(9)));
}

TEST(StatementListGuard, ReplacementStillVisitsNestedBranches) {
  Block root;
  auto *old = root.insert(constant(1));
  auto *replacement = root.insert(constant(2));
  auto *branch = root.push_back<IfStmt>(old)->as<IfStmt>();
  auto yes = std::make_unique<Block>();
  auto *yes_use = yes->push_back<BinaryOpStmt>(BinaryOpType::add, old, old)->as<BinaryOpStmt>();
  auto no = std::make_unique<Block>();
  auto *no_use = no->push_back<BinaryOpStmt>(BinaryOpType::sub, old, old)->as<BinaryOpStmt>();
  branch->set_true_statements(std::move(yes));
  branch->set_false_statements(std::move(no));
  irpass::replace_all_usages_with(&root, old, replacement);
  EXPECT_EQ(branch->cond, replacement);
  EXPECT_EQ(yes_use->lhs, replacement);
  EXPECT_EQ(yes_use->rhs, replacement);
  EXPECT_EQ(no_use->lhs, replacement);
  EXPECT_EQ(no_use->rhs, replacement);
}

}  // namespace
}  // namespace quadrants::lang
