#include "gtest/gtest.h"

#include "quadrants/common/core.h"

namespace quadrants {

TEST(CoreTest, Basic) {
  EXPECT_EQ(trim_string("hello quadrants  "), "hello quadrants");
}

}  // namespace quadrants
