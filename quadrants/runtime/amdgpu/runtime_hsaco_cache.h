#pragma once

#include "quadrants/common/byte_cache.h"

namespace quadrants::lang {

struct RuntimeHsacoKey {
  std::string ir_digest;
  std::string mcpu;
  std::string features;
  bool fast_math;

  bool operator==(const RuntimeHsacoKey &other) const {
    return ir_digest == other.ir_digest && mcpu == other.mcpu && features == other.features &&
           fast_math == other.fast_math;
  }
};

// Cache host bytes only; each JIT session still owns its loaded GPU modules.
using RuntimeHsacoCache = ByteCache<RuntimeHsacoKey>;

}  // namespace quadrants::lang
