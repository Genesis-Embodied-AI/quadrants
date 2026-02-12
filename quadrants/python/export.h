/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

#include "pybind11/pybind11.h"
#include "pybind11/operators.h"
#include "pybind11/stl.h"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "quadrants/common/core.h"

namespace quadrants {

namespace py = pybind11;

using py::literals::operator""_a;

void export_lang(py::module &m);

void export_math(py::module &m);

void export_misc(py::module &m);

}  // namespace quadrants
