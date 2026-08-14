#pragma once

#include <string>

#include "quadrants/rhi/arch.h"

namespace quadrants::lang {

struct CompileConfig;
struct DeviceCapabilityConfig;
class Program;
class IRNode;
class SNode;
class Kernel;

std::string get_hashed_offline_cache_key_of_snode(const SNode *snode);
std::string get_hashed_offline_cache_key(const CompileConfig &config,
                                         const DeviceCapabilityConfig &caps,
                                         Kernel *kernel);
// Stable hex fingerprint of the device capabilities alone, using the same serialization that feeds
// get_hashed_offline_cache_key. Exposed to the Python fast-cache so its checksum can distinguish devices whose caps
// change the generated code (e.g. SPIR-V int64 / atomics families), matching the caps-awareness of the native cache.
std::string get_hashed_offline_cache_key_of_device_caps(const DeviceCapabilityConfig &caps);
void gen_offline_cache_key(IRNode *ast, std::ostream *os);

}  // namespace quadrants::lang
