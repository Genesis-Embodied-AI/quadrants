#pragma once

#include <unordered_set>

namespace quadrants::lang {
class Block;
class IRNode;
class Stmt;

std::unordered_set<Stmt *> gather_forward_statements(IRNode *root);

// Reject execution constructs for which the prototype cannot safely abort a GPU thread on overflow.
void validate_adstack_counting(IRNode *root);
void lower_adstack_forward_count(Block *block, const std::unordered_set<Stmt *> &forward_statements);
void prepare_counted_adstacks(IRNode *root);
}  // namespace quadrants::lang
