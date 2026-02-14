/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "quadrants/common/core.h"
#include "quadrants/common/task.h"
#include "quadrants/util/testing.h"

namespace quadrants {

class RunTests : public Task {
  std::string run(const std::vector<std::string> &parameters) override {
    return std::to_string(run_tests(parameters));
  }
};

QD_IMPLEMENTATION(Task, RunTests, "test");

}  // namespace quadrants
