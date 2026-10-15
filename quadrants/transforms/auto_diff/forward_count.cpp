#include "quadrants/transforms/auto_diff/forward_count.h"
#include "quadrants/transforms/auto_diff/auto_diff_common.h"

namespace quadrants::lang {
namespace {

class GatherForward : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  std::unordered_set<Stmt *> statements;

  GatherForward() {
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    statements.insert(stmt);
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    statements.insert(stmt);
  }
};

class KeepForward : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  const std::unordered_set<Stmt *> &original;

  explicit KeepForward(const std::unordered_set<Stmt *> &statements) : original(statements) {
  }

  void visit(Block *block) override {
    for (int i = static_cast<int>(block->size()) - 1; i >= 0; --i) {
      auto *stmt = block->statements[i].get();
      // MakeAdjoint inserts branch snapshots, and BackupSSA may relocate a stack to its independent block.
      if (!original.count(stmt) && !stmt->is<AdStackAllocaStmt>() && !stmt->is<AdStackPushStmt>()) {
        block->erase(i);
      } else {
        stmt->accept(this);
      }
    }
  }
};

struct Counter {
  Stmt *value;
  Stmt *depth;
  Stmt *peak;
};

}  // namespace

std::unordered_set<Stmt *> gather_forward_statements(IRNode *root) {
  GatherForward gather;
  root->accept(&gather);
  return std::move(gather.statements);
}

void validate_adstack_counting(IRNode *root) {
  for (auto *stmt : gather_forward_statements(root)) {
    QD_ERROR_IF(
        stmt->is<InternalFuncStmt>() || stmt->is<RandStmt>() || stmt->is<WhileStmt>() || stmt->is<ContinueStmt>() ||
            stmt->is<ReturnStmt>() || stmt->is<MeshForStmt>(),
        "Adstack replay counting prototype does not support random, internal/cooperative calls, while, continue, "
        "return, or mesh iteration");
    SNode *field = nullptr;
    if (auto *ptr = stmt->cast<GlobalPtrStmt>())
      field = ptr->snode;
    if (auto *loop = stmt->cast<StructForStmt>())
      field = loop->snode;
    for (auto *node = field; node; node = node->parent) {
      QD_ERROR_IF(node->type != SNodeType::root && node->type != SNodeType::dense && node->type != SNodeType::place,
                  "Adstack replay counting prototype supports dense fields only");
    }
  }
}

void lower_adstack_forward_count(Block *block, const std::unordered_set<Stmt *> &forward_statements) {
  KeepForward keep(forward_statements);
  block->accept(&keep);

  auto statements = irpass::analysis::gather_statements(block, [](Stmt *) { return true; });
  std::unordered_map<Stmt *, Counter> counters;
  for (auto *stmt : statements) {
    if (auto *stack = stmt->cast<AdStackAllocaStmt>()) {
      QD_ERROR_IF(stack->dt->is<TensorType>(), "Adstack replay counting prototype requires scalarized stacks");
      auto *peak = block->insert(Stmt::make<AllocaStmt>(PrimitiveType::i64), 0);
      auto *value = stack->insert_before_me(Stmt::make<AllocaStmt>(stack->dt));
      auto *depth = stack->insert_before_me(Stmt::make<AllocaStmt>(PrimitiveType::i64));
      // Explicit reset at the original allocation point, including reentry from an enclosing loop.
      auto *zero = stack->insert_before_me(Stmt::make<ConstStmt>(TypedConstant(int64_t(0))));
      stack->insert_before_me(Stmt::make<LocalStoreStmt>(depth, zero));
      counters.emplace(stack, Counter{value, depth, peak});
    }
  }

  for (auto *stmt : statements) {
    if (auto *push = stmt->cast<AdStackPushStmt>()) {
      const auto &c = counters.at(push->stack);
      push->insert_before_me(Stmt::make<LocalStoreStmt>(c.value, push->v));
      auto *depth = push->insert_before_me(Stmt::make<LocalLoadStmt>(c.depth));
      auto *one = push->insert_before_me(Stmt::make<ConstStmt>(TypedConstant(int64_t(1))));
      auto *next = push->insert_before_me(Stmt::make<BinaryOpStmt>(BinaryOpType::add, depth, one));
      push->insert_before_me(Stmt::make<LocalStoreStmt>(c.depth, next));
      auto *peak = push->insert_before_me(Stmt::make<LocalLoadStmt>(c.peak));
      auto *maximum = push->insert_before_me(Stmt::make<BinaryOpStmt>(BinaryOpType::max, peak, next));
      push->replace_with(Stmt::make<LocalStoreStmt>(c.peak, maximum));
    } else if (auto *load = stmt->cast<AdStackLoadTopStmt>()) {
      load->replace_with(Stmt::make<LocalLoadStmt>(counters.at(load->stack).value));
    } else {
      QD_ERROR_IF(stmt->is<AdStackPopStmt>() || stmt->is<AdStackLoadTopAdjStmt>() || stmt->is<AdStackAccAdjointStmt>(),
                  "Unexpected reverse operation in adstack counting forward");
    }
  }
  Stmt *maximum = nullptr;
  for (const auto &[stack, counter] : counters) {
    stack->parent->erase(stack);
    auto *peak = block->insert(Stmt::make<LocalLoadStmt>(counter.peak));
    maximum = maximum ? block->insert(Stmt::make<BinaryOpStmt>(BinaryOpType::max, maximum, peak)) : peak;
  }
  // Capacity is shared by all stacks. Combine local peaks before publishing so each iteration only touches
  // the global measurement once, regardless of how many scalar stacks autodiff produced.
  if (maximum)
    block->insert(Stmt::make<InternalFuncStmt>("adstack_measure_peak", std::vector<Stmt *>{maximum}));
}

}  // namespace quadrants::lang
