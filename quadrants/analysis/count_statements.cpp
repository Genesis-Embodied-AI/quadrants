#include "quadrants/ir/ir.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/visitors.h"

namespace quadrants::lang {

// Count all statements (including containers)
class StmtCounter : public BasicStmtVisitor {
 private:
  StmtCounter() {
    counter_ = 0;
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  using BasicStmtVisitor::visit;

 public:
  void preprocess_container_stmt(Stmt *stmt) override {
    counter_++;
  }

  void visit(Stmt *stmt) override {
    counter_++;
  }

  static int run(IRNode *root) {
    StmtCounter stmt_counter;
    root->accept(&stmt_counter);
    return stmt_counter.counter_;
  }

 private:
  int counter_;
};

namespace irpass::analysis {
int count_statements(IRNode *root) {
  QD_ASSERT(root);
  return StmtCounter::run(root);
}
}  // namespace irpass::analysis

}  // namespace quadrants::lang
