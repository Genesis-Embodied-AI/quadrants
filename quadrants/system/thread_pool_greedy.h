#pragma once

#include "quadrants/system/thread_pool_greedy_defaults.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace quadrants {

// Experiment switches are read once at pool construction; absent switches preserve production behavior.
struct ThreadPoolGreedyOptions {
  bool static_tasks = flag("QD_GREEDY_STATIC_TASKS", 1);
  bool caller_poll = flag("QD_GREEDY_CALLER_POLL", 2);
  bool atomic_completion = flag("QD_GREEDY_ATOMIC_COMPLETION", 4);

 private:
  static bool flag(const char *name, int bit) {
    const char *value = std::getenv(name);
    if (!value)
      return (greedy_default_features & bit) != 0;
    if (std::string(value) == "0")
      return false;
    if (std::string(value) == "1")
      return true;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
  }
};

}  // namespace quadrants
