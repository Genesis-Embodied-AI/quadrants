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

  // Number of `qd.checkpoint(...)` blocks (== IF conditional nodes) emitted for the most
  // recent successful graph build / cached launch. `0` for non-CUDA backends (until slice 4/5
  // adds the indirect-dispatch implementation), or for any kernel with no `qd.checkpoint()`
  // blocks. Used by tests to assert the GraphManager actually wired the IF path.
  virtual std::size_t get_graph_num_checkpoints_on_last_call() const {
    return 0;
  }

  virtual std::size_t get_graph_total_builds() const {
    return 0;
  }

  virtual ~KernelLauncher() = default;
};

}  // namespace quadrants::lang
