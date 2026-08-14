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

// Per-offloaded-task cache key. See perso_hugh/doc/quadrants_per_task_ir_key_design_2026jul22.md for the soundness
// contract. Folds the task's CHI IR, compile_config, device caps, the layout signature of every SNode tree the task
// touches, the owning kernel's argument/return ABI (the context struct the task's compiled code reads/writes against),
// and autodiff_mode into one deterministic, collision-safe key. `task` must be the post-`re_id` single-task IR (as
// produced per task in `KernelCodeGen::compile_kernel_to_module`).
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
