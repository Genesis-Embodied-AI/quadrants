#pragma once

#include <vector>

#include <pybind11/pybind11.h>

#include "quadrants/ir/snode.h"

namespace quadrants::lang {
class Program;
class Ndarray;

pybind11::capsule ndarray_to_dlpack(Program *program,
                                    pybind11::object owner,
                                    Ndarray *ndarray,
                                    const std::vector<int> &layout = {},
                                    bool versioned = false);
pybind11::capsule field_to_dlpack(Program *program,
                                  SNode *snode,
                                  int element_ndim,
                                  int n,
                                  int m,
                                  bool versioned = false);
}  // namespace quadrants::lang
