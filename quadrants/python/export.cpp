/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "quadrants/python/export.h"
#include "quadrants/common/interface.h"
#include "quadrants/util/io.h"

namespace quadrants {

PYBIND11_MODULE(quadrants_python, m) {
  m.doc() = "quadrants_python";

  for (auto &kv : InterfaceHolder::get_instance()->methods) {
    kv.second(&m);
  }

  export_lang(m);
  export_math(m);
  export_misc(m);
}

}  // namespace quadrants
