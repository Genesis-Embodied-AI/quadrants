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

// Per-construct cache key. Keys the pre-offload, isolated, re-id'd construct sub-IR (the input to the per-construct
// frontend split) so a cache can reuse an unchanged construct's frontend output (its OffloadedStmt tasks) instead of
// re-running simplify/merge_global_ptrs/offload. Folds compile config, touched-SNode tree ids, kernel argument/return
// ABI, autodiff mode and the construct body IR. It deliberately omits device caps and the O(tree) full-layout hash as
// a cost choice, so a cache consuming this key across processes must fold those itself. This PR defines the key (it
// keys the split, which lives here) but has no live consumer -- the split runs cache-free; the cross-process manifest
// PR is the consumer. `construct` is the isolated construct block AFTER lower_ast + backward-slice isolation + die,
// and must be re-id'd by the caller for determinism.
std::string get_hashed_per_construct_cache_key(const CompileConfig &config,
                                               IRNode *construct,
                                               const Kernel *kernel);

void gen_offline_cache_key(IRNode *ast, std::ostream *os);

}  // namespace quadrants::lang
