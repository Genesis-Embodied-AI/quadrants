#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/system/profiler.h"

namespace quadrants::lang {

namespace {

// Remove all the assume in range statements.
// These statements are useless after make_block_local.
// Their existence harms IR optimization quality.

class RemoveRangeAssumption : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;

  void visit(RangeAssumptionStmt *stmt) override {
    stmt->replace_usages_with(stmt->input);
    modifier.erase(stmt);
  }

  static bool run(IRNode *node) {
    RemoveRangeAssumption pass;
    node->accept(&pass);
    return pass.modifier.modify_ir();
  }
};

}  // namespace

namespace irpass {

bool remove_range_assumption(IRNode *root) {
  TI_AUTO_PROF;
  return RemoveRangeAssumption::run(root);
}

}  // namespace irpass

}  // namespace quadrants::lang
