#pragma once

#include "quadrants/ir/pass.h"

namespace quadrants::lang {

class ConstantFoldPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    Program *program;
  };
};

}  // namespace quadrants::lang
