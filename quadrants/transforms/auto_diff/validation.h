#pragma once

#include <string>

namespace quadrants::lang {

class IRNode;
class Block;

// Stage: Validation. Diagnostic-only passes that refuse autodiff inputs which would silently produce wrong gradients,
// and inserted runtime assertions for global-data access rules.

// Refuses cross-iteration read-after-write through a needs_grad global field at an offload-level loop body. Run
// pre-transform from irpass::auto_diff.
void offload_level_global_cross_iter_raw_check(Block *offload_body);

// Inserts runtime check-bit accumulators around GlobalStore / AtomicOp on snodes whose adjoint check-bit is enabled.
// Run from irpass::differentiation_validation_check.
void global_data_access_rule_check(IRNode *root, const std::string &kernel_name);

}  // namespace quadrants::lang
