/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "quadrants/python/export.h"
#include "quadrants/common/interface.h"
#include "quadrants/util/io.h"

namespace quadrants {

NB_MODULE(quadrants_python, m) {
  m.doc() = "quadrants_python";

  // Disable nanobind's interpreter-shutdown leak checker. It reports the primitive DataType_* module
  // attributes (registered via the PER_TYPE X-macro in export_lang.cpp) plus the class/method objects
  // they keep alive as "leaks". These are long-lived module-global singletons, not a growing leak, so
  // the warnings are pure noise for downstream users (e.g. Genesis) at process exit.
  nb::set_leak_warnings(false);

  register_py_exception_translator();

  for (auto &kv : InterfaceHolder::get_instance()->methods) {
    kv.second(&m);
  }

  export_lang(m);
  export_math(m);
  export_misc(m);
}

}  // namespace quadrants
