#include "quadrants/ir/ir.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/visitors.h"

namespace quadrants::lang {

// Find if there is a store (or AtomicOpStmt).
class LocalStoreSearcher : public BasicStmtVisitor {
 private:
  const std::vector<Stmt *> &vars_;
  bool result_;

 public:
  using BasicStmtVisitor::visit;

  explicit LocalStoreSearcher(const std::vector<Stmt *> &vars)
      : vars_(vars), result_(false) {
    for (auto var : vars) {
      TI_ASSERT(var->is<AllocaStmt>());
    }
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(LocalStoreStmt *stmt) override {
    for (auto var : vars_) {
      if (stmt->dest == var) {
        result_ = true;
        break;
      }
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    for (auto var : vars_) {
      if (stmt->dest == var) {
        result_ = true;
        break;
      }
    }
  }

  static bool run(IRNode *root, const std::vector<Stmt *> &vars) {
    LocalStoreSearcher searcher(vars);
    root->accept(&searcher);
    return searcher.result_;
  }
};

namespace irpass::analysis {
bool has_store_or_atomic(IRNode *root, const std::vector<Stmt *> &vars) {
  return LocalStoreSearcher::run(root, vars);
}
}  // namespace irpass::analysis

}  // namespace quadrants::lang
