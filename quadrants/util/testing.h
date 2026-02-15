/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#define BENCHMARK CATCH_BENCHMARK
#include <catch.hpp>
#undef BENCHMARK

namespace quadrants {

#define QD_CHECK_EQUAL(A, B, tolerance)              \
  {                                                  \
    if (!quadrants::math::equal(A, B, tolerance)) {  \
      std::cout << A << std::endl << B << std::endl; \
    }                                                \
    CHECK(quadrants::math::equal(A, B, tolerance));  \
  }

#define QD_ASSERT_EQUAL(A, B, tolerance)             \
  {                                                  \
    if (!quadrants::math::equal(A, B, tolerance)) {  \
      std::cout << A << std::endl << B << std::endl; \
      QD_ERROR(#A " != " #B);                        \
    }                                                \
  }

#define QD_TEST(x) TEST_CASE(x, ("[" x "]"))
#define QD_CHECK(x) CHECK(x)
#define QD_TEST_PROGRAM                     \
  auto prog_ = std::make_unique<Program>(); \
  prog_->materialize_runtime();

int run_tests(std::vector<std::string> argv);

}  // namespace quadrants
