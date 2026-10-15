#pragma once

#include <unordered_set>

namespace quadrants::lang {
class Block;
class IRNode;
class Stmt;

std::unordered_set<Stmt *> gather_forward_statements(IRNode *root);

// Reject execution constructs outside the prototype's supported counting and replay subset.
void validate_adstack_counting(IRNode *root);
void lower_adstack_forward_count(Block *block, const std::unordered_set<Stmt *> &forward_statements);
}  // namespace quadrants::lang
