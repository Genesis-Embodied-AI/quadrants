#pragma once

#include "quadrants/ir/pass.h"

namespace quadrants::lang {

class MakeMeshThreadLocal : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
  };
};

}  // namespace quadrants::lang
