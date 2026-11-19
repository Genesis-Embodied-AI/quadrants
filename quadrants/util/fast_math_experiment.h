#pragma once

#include <cstdlib>

namespace quadrants::lang {

// Experiment-only controls for issue #1086. Run in fresh processes with all compilation caches disabled.
inline bool experiment_disable_reassociation() {
  const char *value = std::getenv("QD_EXPERIMENT_NO_REASSOC");
  return value != nullptr && value[0] == '1';
}

inline bool experiment_disable_contraction() {
  const char *value = std::getenv("QD_EXPERIMENT_NO_CONTRACT");
  return value != nullptr && value[0] == '1';
}

}  // namespace quadrants::lang
