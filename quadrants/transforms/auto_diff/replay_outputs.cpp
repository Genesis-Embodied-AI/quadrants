#include "quadrants/transforms/auto_diff/forward_count.h"
#include "quadrants/transforms/auto_diff/auto_diff_common.h"

namespace quadrants::lang {
namespace {

GlobalPtrStmt *field_pointer(Stmt *ptr) {
  while (auto *matrix = ptr->cast<MatrixPtrStmt>())
    ptr = matrix->origin;
  return ptr->cast<GlobalPtrStmt>();
}

ArgLoadStmt *array_pointer(Stmt *ptr) {
  while (auto *matrix = ptr->cast<MatrixPtrStmt>())
    ptr = matrix->origin;
  if (auto *external = ptr->cast<ExternalPtrStmt>())
    return external->base_ptr->cast<ArgLoadStmt>();
  return nullptr;
}

}  // namespace

void remove_adstack_count_outputs(IRNode *root, const CompileConfig &config) {
  std::unordered_set<SNode *> written_fields;
  std::set<std::vector<int>> written_arrays;
  // Normal autodiff retains some non-differentiable outputs. Strip their writes, then let dead-code elimination
  // remove computations used only by those outputs, including read/modify/write diagnostic flags. A remaining
  // read of any written buffer could observe a removed write, so reject it conservatively across the whole kernel.
  auto writes = irpass::analysis::gather_statements(
      root, [](Stmt *stmt) { return stmt->is<GlobalStoreStmt>() || stmt->is<AtomicOpStmt>(); });
  if (writes.empty())
    return;
  for (auto *stmt : writes) {
    auto *atomic = stmt->cast<AtomicOpStmt>();
    auto *dest = atomic ? atomic->dest : stmt->as<GlobalStoreStmt>()->dest;
    auto *field = field_pointer(dest);
    auto *array = array_pointer(dest);
    QD_ERROR_IF(!field && !array, "Adstack counting cannot identify a non-differentiable output buffer");
    if (field)
      written_fields.insert(field->snode);
    if (array)
      written_arrays.insert(array->arg_id);
    if (atomic)
      atomic->replace_with(Stmt::make<GlobalLoadStmt>(dest));
    else
      stmt->parent->erase(stmt);
  }
  irpass::full_simplify(root, config, {false, /*autodiff_enabled=*/false});
  for (auto *stmt : irpass::analysis::gather_statements(root, [](Stmt *s) { return s->is<GlobalLoadStmt>(); })) {
    auto *source = stmt->as<GlobalLoadStmt>()->src;
    auto *field = field_pointer(source);
    auto *array = array_pointer(source);
    // External aliases are checked before executing the counting kernel, including persistent-cache hits.
    QD_ERROR_IF((field && written_fields.count(field->snode)) || (array && written_arrays.count(array->arg_id)),
                "Adstack counting replay reads a non-differentiable buffer that it also writes; "
                "counting must not change primal state");
  }
}

}  // namespace quadrants::lang
