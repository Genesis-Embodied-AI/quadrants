#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/system/profiler.h"

namespace quadrants::lang {

namespace {

// Remove all the loop_unique statements.

class RemoveLoopUnique : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;

  void visit(LoopUniqueStmt *stmt) override {
    stmt->replace_usages_with(stmt->input);
    modifier.erase(stmt);
  }

  static bool run(IRNode *node) {
    RemoveLoopUnique pass;
    node->accept(&pass);
    return pass.modifier.modify_ir();
  }
};

}  // namespace

namespace irpass {

bool remove_loop_unique(IRNode *root) {
  QD_AUTO_PROF;
  return RemoveLoopUnique::run(root);
}

}  // namespace irpass

}  // namespace quadrants::lang
