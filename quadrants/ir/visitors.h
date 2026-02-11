#pragma once
#include "quadrants/ir/ir.h"

namespace quadrants::lang {

// Visits all non-containing statements
class BasicStmtVisitor : public IRVisitor {
 public:
  BasicStmtVisitor();

  virtual void preprocess_container_stmt(Stmt *stmt) {
  }

  void visit(Block *stmt_list) override;

  void visit(IfStmt *if_stmt) override;

  void visit(WhileStmt *stmt) override;

  void visit(RangeForStmt *for_stmt) override;

  void visit(StructForStmt *for_stmt) override;

  void visit(MeshForStmt *for_stmt) override;

  void visit(OffloadedStmt *stmt) override;

  void visit(FrontendWhileStmt *stmt) override;

  void visit(FrontendForStmt *stmt) override;

  void visit(FrontendIfStmt *stmt) override;
};

}  // namespace quadrants::lang
