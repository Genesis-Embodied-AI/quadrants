/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "quadrants/python/exception.h"

namespace quadrants {

void raise_assertion_failure_in_python(const std::string &msg) {
  throw ExceptionForPython(msg);
}

}  // namespace quadrants

void quadrants_raise_assertion_failure_in_python(const char *msg) {
  quadrants::raise_assertion_failure_in_python(std::string(msg));
}
