#pragma once

#include "quadrants/ir/pass.h"

namespace quadrants::lang {

class MakeBlockLocalPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
    bool verbose;
  };
};

}  // namespace quadrants::lang
