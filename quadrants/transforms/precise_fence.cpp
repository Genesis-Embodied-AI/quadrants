#include "quadrants/transforms/precise_fence.h"

#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/ir/stmt_op_types.h"
#include "quadrants/ir/type_factory.h"

namespace quadrants::lang {

namespace {

// Encode op types as integers in the func_name to avoid needing name-to-enum reverse lookups.
// Format: "__qd_precise_bin_<int>" or "__qd_precise_un_<int>" (with optional ",<cast_type_id>").
const std::string kPreciseBinPrefix = "__qd_precise_bin_";
const std::string kPreciseUnPrefix = "__qd_precise_un_";

std::string encode_binary(BinaryOpType op) {
  return kPreciseBinPrefix + std::to_string(static_cast<int>(op));
}

std::string encode_unary(UnaryOpType op, const DataType &cast_type) {
  std::string s = kPreciseUnPrefix + std::to_string(static_cast<int>(op));
  if (cast_type != PrimitiveType::unknown) {
    s += "," + std::to_string(static_cast<int>(cast_type->as<PrimitiveType>()->type));
  }
  return s;
}

class FencePreciseOps : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  explicit FencePreciseOps() {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
  }

  void visit(UnaryOpStmt *stmt) override {
    if (!(stmt->codegen_hints & (uint32_t)CodegenHint::kDisableFastMath))
      return;

    std::string name = encode_unary(stmt->op_type, stmt->is_cast() ? stmt->cast_type : PrimitiveType::unknown);

    auto replacement = std::make_unique<InternalFuncStmt>(
        name, std::vector<Stmt *>{stmt->operand}, static_cast<Type *>(stmt->ret_type), /*with_runtime_context=*/false);
    replacement->codegen_hints = stmt->codegen_hints;
    replacement->ret_type = stmt->ret_type;

    stmt->replace_usages_with(replacement.get());
    modifier.insert_before(stmt, std::move(replacement));
    modifier.erase(stmt);
  }

  void visit(BinaryOpStmt *stmt) override {
    if (!(stmt->codegen_hints & (uint32_t)CodegenHint::kDisableFastMath))
      return;

    std::string name = encode_binary(stmt->op_type);

    auto replacement = std::make_unique<InternalFuncStmt>(
        name, std::vector<Stmt *>{stmt->lhs, stmt->rhs}, static_cast<Type *>(stmt->ret_type), /*with_runtime_context=*/false);
    replacement->codegen_hints = stmt->codegen_hints;
    replacement->ret_type = stmt->ret_type;

    stmt->replace_usages_with(replacement.get());
    modifier.insert_before(stmt, std::move(replacement));
    modifier.erase(stmt);
  }

  static bool run(IRNode *root) {
    FencePreciseOps pass;
    root->accept(&pass);
    return pass.modifier.modify_ir();
  }

 private:
  DelayedIRModifier modifier;
};

class UnfencePreciseOps : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  explicit UnfencePreciseOps() {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
  }

  void visit(InternalFuncStmt *stmt) override {
    std::unique_ptr<Stmt> replacement;

    if (stmt->func_name.rfind(kPreciseBinPrefix, 0) == 0) {
      std::string encoded = stmt->func_name.substr(kPreciseBinPrefix.size());
      auto op_type = static_cast<BinaryOpType>(std::stoi(encoded));
      QD_ASSERT(stmt->args.size() == 2);
      auto bin = std::make_unique<BinaryOpStmt>(op_type, stmt->args[0], stmt->args[1]);
      bin->ret_type = stmt->ret_type;
      bin->codegen_hints = stmt->codegen_hints;
      replacement = std::move(bin);
    } else if (stmt->func_name.rfind(kPreciseUnPrefix, 0) == 0) {
      std::string encoded = stmt->func_name.substr(kPreciseUnPrefix.size());
      DataType cast_type = PrimitiveType::unknown;
      auto comma = encoded.find(',');
      int op_int;
      if (comma != std::string::npos) {
        op_int = std::stoi(encoded.substr(0, comma));
        int cast_int = std::stoi(encoded.substr(comma + 1));
        cast_type = TypeFactory::get_instance().get_primitive_type(static_cast<PrimitiveTypeID>(cast_int));
      } else {
        op_int = std::stoi(encoded);
      }
      auto op_type = static_cast<UnaryOpType>(op_int);
      QD_ASSERT(stmt->args.size() == 1);
      auto un = std::make_unique<UnaryOpStmt>(op_type, stmt->args[0]);
      un->ret_type = stmt->ret_type;
      un->cast_type = cast_type;
      un->codegen_hints = stmt->codegen_hints;
      replacement = std::move(un);
    } else {
      return;
    }

    stmt->replace_usages_with(replacement.get());
    modifier.insert_before(stmt, std::move(replacement));
    modifier.erase(stmt);
  }

  static bool run(IRNode *root) {
    UnfencePreciseOps pass;
    root->accept(&pass);
    return pass.modifier.modify_ir();
  }

 private:
  DelayedIRModifier modifier;
};

}  // namespace

namespace irpass {

void fence_precise_ops(IRNode *root) {
  FencePreciseOps::run(root);
}

void unfence_precise_ops(IRNode *root) {
  UnfencePreciseOps::run(root);
}

}  // namespace irpass
}  // namespace quadrants::lang
