#include <csignal>
#include <string>

#include <nanobind/nanobind.h>

#include "quadrants/python/export.h"

namespace quadrants {

// Translates and propagates a CPP exception to Python.
//
// Unlike pybind11, nanobind does not support calling its API (including register_exception_translator) from a
// global/static constructor: doing so runs before the extension's NB_MODULE init has set up nanobind's internals
// and crashes during dlopen. So this is now an explicit function invoked from NB_MODULE (see export.cpp).
void register_py_exception_translator() {
  nanobind::register_exception_translator([](const std::exception_ptr &p, void * /*payload*/) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const std::string &e) {
      PyErr_SetString(PyExc_RuntimeError, e.c_str());
    }
  });
}

}  // namespace quadrants
