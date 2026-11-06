#include "gtest/gtest.h"

#include "quadrants/ir/analysis.h"
#include "quadrants/ir/ir_builder.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace quadrants::lang {

namespace {

int count_stmts(Block *block, const std::function<bool(Stmt *)> &pred) {
  int n = 0;
  for (auto &s : block->statements) {
    if (pred(s.get())) {
      n++;
    }
  }
  return n;
}

bool is_const_i32(Stmt *s, int32 value) {
  auto *c = s->cast<ConstStmt>();
  return c != nullptr && c->val.val_i32 == value;
}

}  // namespace

// merge_global_ptrs (pointers-only WholeKernelCSE): every user of an eliminated statement is rewritten wherever it
// is - nested loop/branch bodies, operands of container statements, users that only become eliminable after an
// earlier rewrite - while sibling scopes are never merged into each other.
TEST(WholeKernelCSE, MergeGlobalPtrsRewritesUsersAcrossScopes) {
  TestProgram test_prog;
  test_prog.setup();

  IRBuilder b;
  auto *zero_a = b.get_int32(0);
  auto *zero_b = b.get_int32(0);  // duplicate root constant; only used as loop2's begin (a container operand)
  auto *n = b.get_int32(8);
  auto *one_a = b.get_int32(1);
  auto *one_b = b.get_int32(1);  // duplicate root constant
  auto *arr = b.create_ndarray_arg_load({0}, PrimitiveType::i32, 1);

  auto *loop1 = b.create_range_for(zero_a, n);
  Stmt *x, *p, *k5_loop1;
  IfStmt *branch;
  {
    auto _ = b.get_loop_guard(loop1);
    auto *i = b.get_loop_index(loop1, 0);
    auto *one_c = b.get_int32(1);      // nested duplicate of a visible root constant
    x = b.create_add(i, one_b);        // one_b -> one_a
    auto *y = b.create_add(i, one_c);  // one_c -> one_a, then y -> x
    p = b.create_external_ptr(arr, {x});
    auto *q = b.create_external_ptr(arr, {y});  // same address once y -> x
    b.create_global_store(q, i);
    k5_loop1 = b.get_int32(5);
    branch = b.create_if(b.create_cmp_lt(i, k5_loop1));
    {
      auto _ = b.get_if_guard(branch, true);
      auto *z = b.create_add(i, one_c);           // -> x
      auto *r = b.create_external_ptr(arr, {z});  // -> p
      b.create_global_store(r, one_c);
    }
  }
  auto *loop2 = b.create_range_for(zero_b, n);
  Stmt *k5_loop2;
  {
    auto _ = b.get_loop_guard(loop2);
    auto *j = b.get_loop_index(loop2, 0);
    k5_loop2 = b.get_int32(5);  // same value as loop1's constant, but in a sibling scope
    auto *w = b.create_add(j, k5_loop2);
    b.create_global_store(b.create_external_ptr(arr, {w}), j);
  }

  auto ir = b.extract_ir();
  auto *root = ir->as<Block>();
  irpass::type_check(ir.get(), test_prog.prog()->compile_config());
  EXPECT_TRUE(irpass::merge_global_ptrs(ir.get()));
  irpass::analysis::verify(ir.get());

  // Root: one constant 0 and one constant 1 remain; loop2's begin was rewritten to the kept constant.
  EXPECT_EQ(count_stmts(root, [](Stmt *s) { return is_const_i32(s, 0); }), 1);
  EXPECT_EQ(count_stmts(root, [](Stmt *s) { return is_const_i32(s, 1); }), 1);
  EXPECT_EQ(loop2->as<RangeForStmt>()->begin, zero_a);
  EXPECT_EQ(root->locate(zero_b), -1);

  // loop1 body: one add, one pointer; the plain store and the in-branch store both write through it.
  auto *body1 = loop1->as<RangeForStmt>()->body.get();
  EXPECT_EQ(
      count_stmts(body1,
                  [](Stmt *s) { return s->is<BinaryOpStmt>() && s->as<BinaryOpStmt>()->op_type == BinaryOpType::add; }),
      1);
  EXPECT_EQ(count_stmts(body1, [](Stmt *s) { return s->is<ExternalPtrStmt>(); }), 1);
  EXPECT_EQ(count_stmts(body1, [](Stmt *s) { return is_const_i32(s, 1); }), 0);
  EXPECT_EQ(x->as<BinaryOpStmt>()->rhs, one_a);
  for (auto &s : body1->statements) {
    if (auto *st = s->cast<GlobalStoreStmt>()) {
      EXPECT_EQ(st->dest, p);
    }
  }
  auto *then_block = branch->true_statements.get();
  ASSERT_NE(then_block, nullptr);
  ASSERT_EQ(then_block->size(), 1);
  auto *inner_store = then_block->statements[0]->as<GlobalStoreStmt>();
  EXPECT_EQ(inner_store->dest, p);
  EXPECT_EQ(inner_store->val, one_a);

  // Sibling scopes are not merged: each loop keeps its own constant 5.
  auto *body2 = loop2->as<RangeForStmt>()->body.get();
  EXPECT_NE(body1->locate(k5_loop1), -1);
  EXPECT_NE(body2->locate(k5_loop2), -1);
}

// Full WholeKernelCSE: hoisting an if's common first statement changes the tree mid-round; eliminations after it in
// the same round must still rewrite every user and the result must verify.
TEST(WholeKernelCSE, FullModeHoistThenEliminate) {
  TestProgram test_prog;
  test_prog.setup();

  IRBuilder b;
  auto *zero = b.get_int32(0);
  auto *n = b.get_int32(8);
  auto *arr = b.create_ndarray_arg_load({0}, PrimitiveType::i32, 1);
  auto *loop = b.create_range_for(zero, n);
  Stmt *after_a, *after_b;
  {
    auto _ = b.get_loop_guard(loop);
    auto *i = b.get_loop_index(loop, 0);
    auto *branch = b.create_if(b.create_cmp_lt(i, b.get_int32(4)));
    {
      auto _ = b.get_if_guard(branch, true);
      auto *t = b.create_mul(i, b.get_int32(3));
      b.create_global_store(b.create_external_ptr(arr, {t}), t);
    }
    {
      auto _ = b.get_if_guard(branch, false);
      auto *f = b.create_mul(i, b.get_int32(3));
      b.create_global_store(b.create_external_ptr(arr, {f}), i);
    }
    after_a = b.create_add(i, b.get_int32(2));
    after_b = b.create_add(i, b.get_int32(2));  // -> after_a
    b.create_global_store(b.create_external_ptr(arr, {after_a}), after_b);
  }

  auto ir = b.extract_ir();
  irpass::type_check(ir.get(), test_prog.prog()->compile_config());
  EXPECT_TRUE(irpass::whole_kernel_cse(ir.get()));
  irpass::analysis::verify(ir.get());

  auto *body = loop->as<RangeForStmt>()->body.get();
  int adds = 0;
  for (auto &s : body->statements) {
    if (auto *bin = s->cast<BinaryOpStmt>(); bin != nullptr && bin->op_type == BinaryOpType::add) {
      adds++;
    }
    if (auto *st = s->cast<GlobalStoreStmt>()) {
      EXPECT_EQ(st->val, after_a);
    }
  }
  EXPECT_EQ(adds, 1);
}

// Exercise both an index built by an earlier elimination and one first built by hoisting. Each branch hoists a
// prefix constant and a suffix store; the second branch repeats this while the first branch's insertions are pending.
class WholeKernelCSEHoistIndex : public ::testing::TestWithParam<bool> {};

TEST_P(WholeKernelCSEHoistIndex, MultipleHoistsRewriteScalarAndContainerUsers) {
  TestProgram test_prog;
  test_prog.setup();
  IRBuilder b;
  auto *zero = b.get_int32(0);
  if (GetParam()) {
    b.get_int32(0);  // Build the index through elimination before the first hoist.
  }
  auto *n = b.get_int32(8);
  auto *arr = b.create_ndarray_arg_load({0}, PrimitiveType::i32, 1);
  auto *out = b.create_external_ptr(arr, {zero});
  auto *loop = b.create_range_for(zero, n);
  std::vector<BinaryOpStmt *> users;
  std::vector<RangeForStmt *> nested_loops;
  std::vector<GlobalStoreStmt *> suffix_stores;
  {
    auto guard = b.get_loop_guard(loop);
    auto *i = b.get_loop_index(loop);
    auto *cond = b.create_cmp_lt(i, n);
    for (int branch_index = 0; branch_index < 2; ++branch_index) {
      auto *branch = b.create_if(cond);
      {
        auto guard = b.get_if_guard(branch, true);
        auto *three = b.get_int32(3);
        auto *value = b.create_mul(i, three);
        users.push_back(value);
        b.create_global_store(out, value);
        b.create_global_store(out, i);  // Shared suffix, hoisted after the if.
        suffix_stores.push_back(branch->true_statements->statements.back()->as<GlobalStoreStmt>());
      }
      {
        auto guard = b.get_if_guard(branch, false);
        auto *three = b.get_int32(3);
        auto *value = b.create_add(three, three);  // One user occurs twice in the collected list.
        users.push_back(value);
        auto *nested = b.create_range_for(three, n);  // A container operand uses the discarded prefix.
        nested_loops.push_back(nested);
        {
          auto guard = b.get_loop_guard(nested);
          b.create_global_store(out, value);
        }
        b.create_global_store(out, i);
      }
    }
    auto *five_a = b.get_int32(5);
    auto *five_b = b.get_int32(5);
    b.create_global_store(out, b.create_add(five_a, five_b));  // Eliminate again after both hoists.
  }

  auto ir = b.extract_ir();
  irpass::type_check(ir.get(), test_prog.prog()->compile_config());
  EXPECT_TRUE(irpass::whole_kernel_cse(ir.get()));
  irpass::analysis::verify(ir.get());

  auto *body = loop->body.get();
  ASSERT_EQ(count_stmts(body, [](Stmt *stmt) { return is_const_i32(stmt, 3); }), 1);
  Stmt *three = nullptr;
  for (auto &stmt : body->statements) {
    if (is_const_i32(stmt.get(), 3)) {
      three = stmt.get();
    }
  }
  for (auto *user : users) {
    EXPECT_EQ(user->rhs, three);
    if (user->op_type == BinaryOpType::add) {
      EXPECT_EQ(user->lhs, three);
    }
  }
  for (auto *nested : nested_loops) {
    EXPECT_EQ(nested->begin, three);
  }
  for (auto *store : suffix_stores) {
    EXPECT_EQ(store->parent, body);
    EXPECT_NE(body->locate(store), -1);
  }
  EXPECT_EQ(count_stmts(body, [](Stmt *stmt) { return is_const_i32(stmt, 5); }), 1);
  EXPECT_FALSE(irpass::whole_kernel_cse(ir.get()));
}

INSTANTIATE_TEST_SUITE_P(IndexBuiltBeforeOrDuringHoisting, WholeKernelCSEHoistIndex, ::testing::Bool());

// Moving a whole container preserves the indexed operands in its body, even while insertion is queued. The
// discarded container's descendants remain alive in the block's trash bin until the next round clears the index.
TEST(WholeKernelCSE, HoistedContainersKeepNestedUsersValid) {
  TestProgram test_prog;
  test_prog.setup();
  IRBuilder b;
  auto *zero = b.get_int32(0);
  b.get_int32(0);  // Build the index before hoisting the container.
  auto *n = b.get_int32(8);
  auto *arr = b.create_ndarray_arg_load({0}, PrimitiveType::i32, 1);
  auto *out = b.create_external_ptr(arr, {zero});
  auto *branch = b.create_if(b.create_arg_load({1}, PrimitiveType::i32, false));
  RangeForStmt *retained_loop = nullptr;
  for (bool true_branch : {true, false}) {
    auto guard = b.get_if_guard(branch, true_branch);
    auto *loop = b.create_range_for(zero, n);
    if (true_branch) {
      retained_loop = loop;
    }
    auto loop_guard = b.get_loop_guard(loop);
    auto *i = b.get_loop_index(loop);
    auto *one_a = b.get_int32(1);
    auto *one_b = b.get_int32(1);
    b.create_global_store(out, b.create_add(i, b.create_add(one_a, one_b)));
  }
  auto *two_a = b.get_int32(2);
  auto *two_b = b.get_int32(2);
  auto *after = b.create_add(two_a, two_b);
  b.create_global_store(out, after);

  auto ir = b.extract_ir();
  irpass::type_check(ir.get(), test_prog.prog()->compile_config());
  EXPECT_TRUE(irpass::whole_kernel_cse(ir.get()));
  irpass::analysis::verify(ir.get());

  EXPECT_EQ(retained_loop->parent, ir.get());
  EXPECT_EQ(count_stmts(ir.get(), [](Stmt *stmt) { return stmt->is<RangeForStmt>(); }), 1);
  EXPECT_EQ(count_stmts(retained_loop->body.get(), [](Stmt *stmt) { return is_const_i32(stmt, 1); }), 1);
  EXPECT_EQ(after->rhs, two_a);
  EXPECT_FALSE(irpass::whole_kernel_cse(ir.get()));
}

}  // namespace quadrants::lang
