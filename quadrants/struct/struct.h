// Codegen for the hierarchical data structure
#pragma once

#include "quadrants/ir/snode.h"

namespace quadrants::lang {

class StructCompiler {
 public:
  std::vector<SNode *> stack;
  std::vector<SNode *> snodes;
  std::size_t root_size{0};

  virtual ~StructCompiler() = default;

  void collect_snodes(SNode &snode);

  // generate C++/llvm IR
  virtual void generate_types(SNode &snode) = 0;

  virtual void generate_child_accessors(SNode &snode) = 0;

  virtual void run(SNode &node) = 0;
};

}  // namespace quadrants::lang
