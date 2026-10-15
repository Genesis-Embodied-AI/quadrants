#include "quadrants/transforms/auto_diff/forward_count.h"
#include "quadrants/transforms/auto_diff/auto_diff_common.h"

namespace quadrants::lang {
namespace {

struct UniformIndex {
  std::vector<int> argument;
  uint32_t offset;
};

// Recognize an i32 scalar argument plus a constant, or a constant alone. These indices have the same value in
// every iteration and thread. In particular, frame f and frame f+1 are disjoint; particle i and i+1 are not.
std::optional<UniformIndex> uniform_index(Stmt *stmt, int budget = 16) {
  if (!budget || stmt->ret_type != PrimitiveType::i32)
    return std::nullopt;
  if (auto *constant = stmt->cast<ConstStmt>())
    return UniformIndex{{}, static_cast<uint32_t>(constant->val.val_i32)};
  if (auto *argument = stmt->cast<ArgLoadStmt>(); argument && !argument->is_ptr)
    return UniformIndex{argument->arg_id, 0};
  if (auto *binary = stmt->cast<BinaryOpStmt>()) {
    auto lhs = uniform_index(binary->lhs, budget - 1);
    auto rhs = uniform_index(binary->rhs, budget - 1);
    if (lhs && rhs && rhs->argument.empty()) {
      if (binary->op_type == BinaryOpType::add)
        return UniformIndex{lhs->argument, lhs->offset + rhs->offset};
      if (binary->op_type == BinaryOpType::sub)
        return UniformIndex{lhs->argument, lhs->offset - rhs->offset};
    }
  }
  return std::nullopt;
}

struct Access {
  SNode *field{nullptr};
  std::vector<int> array;
  std::vector<std::optional<UniformIndex>> indices;
};

Access access(Stmt *ptr) {
  while (auto *matrix = ptr->cast<MatrixPtrStmt>())
    ptr = matrix->origin;
  Access result;
  std::vector<Stmt *> indices;
  if (auto *field = ptr->cast<GlobalPtrStmt>()) {
    result.field = field->snode;
    indices = field->indices;
  } else if (auto *array = ptr->cast<ExternalPtrStmt>()) {
    result.array = array->base_ptr->as<ArgLoadStmt>()->arg_id;
    indices = array->indices;
  } else {
    QD_ERROR("Adstack counting cannot identify a non-differentiable buffer");
  }
  for (auto *index : indices)
    result.indices.push_back(uniform_index(index));
  return result;
}

bool may_overlap(const Access &a, const Access &b) {
  if (a.field != b.field || a.array != b.array)
    return false;
  if (a.indices.size() == b.indices.size()) {
    for (size_t i = 0; i < a.indices.size(); ++i) {
      const auto &x = a.indices[i];
      const auto &y = b.indices[i];
      if (x && y && x->argument == y->argument && x->offset != y->offset)
        return false;
    }
  }
  return true;
}

}  // namespace

void remove_adstack_count_outputs(IRNode *root, const CompileConfig &config) {
  // Simplify counter locals so frame indices are expressed directly in terms of scalar arguments.
  irpass::full_simplify(root, config, {false, /*autodiff_enabled=*/false});
  auto writes = irpass::analysis::gather_statements(
      root, [](Stmt *stmt) { return stmt->is<GlobalStoreStmt>() || stmt->is<AtomicOpStmt>(); });
  if (writes.empty())
    return;
  std::vector<Access> written;
  // Normal autodiff retains some non-differentiable outputs. Strip their writes, then prune computations used
  // only by those outputs, including read/modify/write diagnostic flags. Save descriptors, not IR pointers:
  // simplification can delete or replace the original address/index statements.
  for (auto *stmt : writes) {
    auto *atomic = stmt->cast<AtomicOpStmt>();
    auto *dest = atomic ? atomic->dest : stmt->as<GlobalStoreStmt>()->dest;
    written.push_back(access(dest));
    if (atomic)
      atomic->replace_with(Stmt::make<GlobalLoadStmt>(dest));
    else
      stmt->parent->erase(stmt);
  }
  irpass::full_simplify(root, config, {false, /*autodiff_enabled=*/false});
  for (auto *stmt : irpass::analysis::gather_statements(root, [](Stmt *s) { return s->is<GlobalLoadStmt>(); })) {
    auto read = access(stmt->as<GlobalLoadStmt>()->src);
    for (const auto &write : written) {
      // External aliases are checked before executing the counting kernel, including persistent-cache hits.
      QD_ERROR_IF(may_overlap(read, write),
                  "Adstack counting replay reads a non-differentiable buffer that it also writes; "
                  "counting must not change primal state");
    }
  }
}

}  // namespace quadrants::lang
