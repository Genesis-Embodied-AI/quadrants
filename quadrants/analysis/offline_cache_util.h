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
class OffloadedStmt;

std::string get_hashed_offline_cache_key_of_snode(const SNode *snode);
std::string get_hashed_offline_cache_key(const CompileConfig &config,
                                         const DeviceCapabilityConfig &caps,
                                         Kernel *kernel);

// Per-offloaded-task cache key: folds the task's CHI IR, compile config, device caps, the layout of every SNode tree
// the task touches, the owning kernel's argument/return ABI, and autodiff mode. Deliberately omits the kernel NAME so
// byte-identical tasks in different kernels share one artifact -- which is why it must never key a name-bearing code
// cache (PTX defines an entry symbol named after its kernel); its sole consumer is the per-task artifact cache, which
// carries the names in its launch metadata. `task` must be the post-`re_id` single-task IR.
std::string get_hashed_per_task_cache_key(const CompileConfig &config,
                                          const DeviceCapabilityConfig &caps,
                                          OffloadedStmt *task,
                                          const Kernel *kernel);
// Stable hex fingerprint of the device capabilities alone, using the same serialization that feeds
// get_hashed_offline_cache_key. Exposed to the Python fast-cache so its checksum can distinguish devices whose caps
// change the generated code (e.g. SPIR-V int64 / atomics families), matching the caps-awareness of the native cache.
std::string get_hashed_offline_cache_key_of_device_caps(const DeviceCapabilityConfig &caps);
void gen_offline_cache_key(IRNode *ast, std::ostream *os);

}  // namespace quadrants::lang
