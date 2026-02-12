#pragma once

#include "quadrants/ir/pass.h"

namespace quadrants::lang {

class LowerAccessPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::vector<SNode *> kernel_forces_no_activate;
    bool lower_atomic;
  };
};

}  // namespace quadrants::lang
