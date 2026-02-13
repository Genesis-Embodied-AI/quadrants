#pragma once

#include "quadrants/ir/pass.h"

namespace quadrants::lang {

class CheckOutOfBoundPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::string kernel_name;
  };
};

}  // namespace quadrants::lang
