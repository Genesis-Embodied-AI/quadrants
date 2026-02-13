#pragma once

#include <pybind11/pybind11.h>

#include "quadrants/ir/snode.h"

namespace quadrants::lang {
class Program;
class Ndarray;

pybind11::capsule ndarray_to_dlpack(Program *program,
                                    pybind11::object owner,
                                    Ndarray *ndarray);
pybind11::capsule field_to_dlpack(Program *program,
                                  SNode *snode,
                                  int element_ndim,
                                  int n,
                                  int m);
}  // namespace quadrants::lang
