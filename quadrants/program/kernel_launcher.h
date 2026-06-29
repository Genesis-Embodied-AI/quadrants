#pragma once

#include "quadrants/codegen/compiled_kernel_data.h"
#include "quadrants/program/launch_context_builder.h"

namespace quadrants::lang {

class KernelLauncher {
 public:
  using Handle = KernelLaunchHandle;

  virtual void launch_kernel(const CompiledKernelData &compiled_kernel_data, LaunchContextBuilder &ctx) = 0;

  virtual std::size_t get_graph_cache_size() const {
    return 0;
  }

  virtual bool get_graph_cache_used_on_last_call() const {
    return false;
  }

  virtual std::size_t get_graph_num_nodes_on_last_call() const {
    return 0;
  }

  // Number of `qd.checkpoint(...)` blocks (== IF conditional nodes) emitted for the most recent successful graph build
  // / cached launch. `0` for non-CUDA backends (until slice 4/5 adds the indirect-dispatch implementation), or for any
  // kernel with no `qd.checkpoint()` blocks. Used by tests to assert the GraphManager actually wired the IF path.
  virtual std::size_t get_graph_num_checkpoints_on_last_call() const {
    return 0;
  }

  // cp_id of the checkpoint that yielded on the most recent graph launch, or `-1` if no checkpoint yielded (this also
  // covers non-graph launches, non-CUDA backends, and kernels without any `qd.checkpoint(yield_on=...)`). Slice 1d test
  // introspection only; slice 2's `GraphStatus` host API supersedes this for end users.
  virtual int get_graph_last_yield_cp_id_on_last_call() const {
    return -1;
  }

  virtual std::size_t get_graph_total_builds() const {
    return 0;
  }

  virtual ~KernelLauncher() = default;
};

}  // namespace quadrants::lang
