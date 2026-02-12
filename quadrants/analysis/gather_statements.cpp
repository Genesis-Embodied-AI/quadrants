#include "quadrants/ir/ir.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/visitors.h"

namespace quadrants::lang {

class StmtSearcher : public BasicStmtVisitor {
 private:
  std::function<bool(Stmt *)> test_;
  std::vector<Stmt *> results_;

 public:
  using BasicStmtVisitor::visit;

  explicit StmtSearcher(std::function<bool(Stmt *)> test) : test_(test) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    if (test_(stmt))
      results_.push_back(stmt);
  }

  static std::vector<Stmt *> run(IRNode *root,
                                 const std::function<bool(Stmt *)> &test) {
    StmtSearcher searcher(test);
    root->accept(&searcher);
    return searcher.results_;
  }
};

namespace irpass::analysis {
std::vector<Stmt *> gather_statements(IRNode *root,
                                      const std::function<bool(Stmt *)> &test) {
  return StmtSearcher::run(root, test);
}
}  // namespace irpass::analysis

}  // namespace quadrants::lang
