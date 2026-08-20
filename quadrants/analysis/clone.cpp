#include "quadrants/ir/ir.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/program/program.h"

#include <unordered_map>

namespace quadrants::lang {

class IRCloner : public IRVisitor {
 private:
  IRNode *other_node;
  std::unordered_map<Stmt *, Stmt *> operand_map_;

 public:
  enum Phase { register_operand_map, replace_operand } phase;

  explicit IRCloner(IRNode *other_node) : other_node(other_node), phase(register_operand_map) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Block *stmt_list) override {
    auto other = other_node->as<Block>();
    for (int i = 0; i < (int)stmt_list->size(); i++) {
      other_node = other->statements[i].get();
      stmt_list->statements[i]->accept(this);
    }
    other_node = other;
  }

  void generic_visit(Stmt *stmt) {
    if (phase == register_operand_map)
      operand_map_[stmt] = other_node->as<Stmt>();
    else {
      QD_ASSERT(phase == replace_operand);
      auto other_stmt = other_node->as<Stmt>();
      QD_ASSERT(stmt->num_operands() == other_stmt->num_operands());
      for (int i = 0; i < stmt->num_operands(); i++) {
        if (operand_map_.find(stmt->operand(i)) == operand_map_.end())
          other_stmt->set_operand(i, stmt->operand(i));
        else
          other_stmt->set_operand(i, operand_map_[stmt->operand(i)]);
      }
    }
  }

  void visit(Stmt *stmt) override {
    generic_visit(stmt);
  }

  void visit(IfStmt *stmt) override {
    generic_visit(stmt);
    auto other = other_node->as<IfStmt>();
    if (stmt->true_statements) {
      other_node = other->true_statements.get();
      stmt->true_statements->accept(this);
      other_node = other;
    }
    if (stmt->false_statements) {
      other_node = other->false_statements.get();
      stmt->false_statements->accept(this);
      other_node = other;
    }
  }

  void visit(WhileStmt *stmt) override {
    generic_visit(stmt);
    auto other = other_node->as<WhileStmt>();
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(RangeForStmt *stmt) override {
    generic_visit(stmt);
    auto other = other_node->as<RangeForStmt>();
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(StructForStmt *stmt) override {
    generic_visit(stmt);
    auto other = other_node->as<StructForStmt>();
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(OffloadedStmt *stmt) override {
    generic_visit(stmt);
    auto other = other_node->as<OffloadedStmt>();

#define CLONE_BLOCK(B)                    \
  if (stmt->B) {                          \
    other->B = std::make_unique<Block>(); \
    other_node = other->B.get();          \
    stmt->B->accept(this);                \
  }

    CLONE_BLOCK(tls_prologue)
    CLONE_BLOCK(bls_prologue)
    CLONE_BLOCK(mesh_prologue)

    if (stmt->body) {
      other_node = other->body.get();
      stmt->body->accept(this);
    }

    CLONE_BLOCK(bls_epilogue)
    CLONE_BLOCK(tls_epilogue)
#undef CLONE_BLOCK

    other_node = other;
  }

  void set_other(IRNode *n) {
    other_node = n;
  }

  static std::unique_ptr<IRNode> run(IRNode *root) {
    std::unique_ptr<IRNode> new_root = root->clone();
    IRCloner cloner(new_root.get());
    cloner.phase = IRCloner::register_operand_map;
    root->accept(&cloner);
    cloner.phase = IRCloner::replace_operand;
    root->accept(&cloner);

    return new_root;
  }
};

namespace irpass::analysis {
std::unique_ptr<IRNode> clone(IRNode *root) {
  return IRCloner::run(root);
}

std::unique_ptr<Block> clone_block_subset(Block *block, const std::vector<int> &indices) {
  // Clone only the listed top-level statements (in the given order) instead of the whole block. The per-construct
  // frontend split needs one isolated copy per construct, and cloning the entire block each time is
  // O(constructs x block size) -- ~5.9 s on a 130-construct / 2685-statement kernel, paid even when every construct
  // is a cache hit. The slice a construct actually needs is tiny by comparison.
  //
  // Operand remapping matches whole-block clone semantics: references between cloned statements are redirected to the
  // clones, while a reference to a statement outside the subset keeps pointing at the original (`generic_visit`'s
  // not-found branch). Callers must therefore pass a subset that is closed under operands if they need a
  // self-contained block.
  auto nb = std::make_unique<Block>();
  nb->set_parent_callable(block->parent_callable());
  std::vector<Stmt *> srcs;
  srcs.reserve(indices.size());
  for (int i : indices) {
    Stmt *s = block->statements[i].get();
    srcs.push_back(s);
    nb->insert(s->clone());
  }
  // Same two-phase walk as IRCloner::run, but driven per statement pair because the source and target blocks no
  // longer line up index-for-index.
  IRCloner cloner(nb.get());
  for (int phase = 0; phase < 2; phase++) {
    cloner.phase = (phase == 0) ? IRCloner::register_operand_map : IRCloner::replace_operand;
    for (std::size_t j = 0; j < srcs.size(); j++) {
      cloner.set_other(nb->statements[j].get());
      srcs[j]->accept(&cloner);
    }
  }
  return nb;
}

std::unique_ptr<Stmt> clone(Stmt *root) {
  auto ret = IRCloner::run(root);
  Stmt *stmt_ptr = dynamic_cast<Stmt *>(ret.release());
  QD_ASSERT(stmt_ptr != nullptr);

  return std::unique_ptr<Stmt>(stmt_ptr);
}
}  // namespace irpass::analysis

}  // namespace quadrants::lang
