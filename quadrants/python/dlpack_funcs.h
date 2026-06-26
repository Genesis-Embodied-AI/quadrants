#pragma once

#include <vector>

#include <nanobind/nanobind.h>

#include "quadrants/ir/snode.h"

namespace quadrants::lang {
class Program;
class Ndarray;

nanobind::capsule ndarray_to_dlpack(Program *program,
                                    nanobind::object owner,
                                    Ndarray *ndarray,
                                    const std::vector<int> &layout = {},
                                    bool versioned = false);
nanobind::capsule field_to_dlpack(Program *program,
                                  SNode *snode,
                                  int element_ndim,
                                  int n,
                                  int m,
                                  bool versioned = false);
}  // namespace quadrants::lang
