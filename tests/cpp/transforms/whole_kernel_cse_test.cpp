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

}  // namespace quadrants::lang
