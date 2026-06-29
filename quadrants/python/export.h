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

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/function.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "quadrants/common/core.h"

namespace quadrants::lang {
class Program;
}  // namespace quadrants::lang

namespace quadrants {

namespace nb = nanobind;

using namespace nb::literals;

void export_lang(nb::module_ &m);

void export_math(nb::module_ &m);

void export_misc(nb::module_ &m);

void export_stream(nb::module_ &m, nb::class_<lang::Program> &program_class);

// Registers the C++ (std::string) -> Python exception translator. Must be called from NB_MODULE rather than a
// static initializer: nanobind's API cannot be used before module init has set up its internals.
void register_py_exception_translator();

}  // namespace quadrants
