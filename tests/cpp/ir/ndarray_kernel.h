#pragma once
#include "quadrants/ir/ir_builder.h"
#include "quadrants/ir/statements.h"
#include "quadrants/inc/constants.h"
#include "quadrants/program/program.h"

namespace quadrants::lang {

std::unique_ptr<Kernel> setup_kernel1(Program *prog);

std::unique_ptr<Kernel> setup_kernel2(Program *prog);
}  // namespace quadrants::lang
