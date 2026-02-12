#pragma once

#include <memory>

#include "quadrants/program/program.h"

namespace quadrants::lang {

class TestProgram {
 public:
  void setup(Arch arch = Arch::x64);

  Program *prog() {
    return prog_.get();
  }

 private:
  std::unique_ptr<Program> prog_{nullptr};
};

}  // namespace quadrants::lang
