#pragma once

#include "quadrants/codegen/compiled_kernel_data.h"
#include "quadrants/program/launch_context_builder.h"

namespace quadrants::lang {

class KernelLauncher {
 public:
  using Handle = KernelLaunchHandle;

  virtual void launch_kernel(const CompiledKernelData &compiled_kernel_data, LaunchContextBuilder &ctx) = 0;

  // Register a compiled kernel without launching it and return its launch_id (stable handle into the launcher's
  // per-kernel state). Used to make a child kernel's compiled artifacts reachable so a graph=True parent can embed
  // it as a subgraph. Returns -1 on backends that do not support nested subgraph launches.
  virtual int ensure_registered(const CompiledKernelData &compiled_kernel_data) {
    return -1;
  }

  virtual std::size_t get_graph_cache_size() const {
    return 0;
  }

  virtual bool get_graph_cache_used_on_last_call() const {
    return false;
  }

  virtual std::size_t get_graph_num_nodes_on_last_call() const {
    return 0;
  }

  virtual std::size_t get_graph_total_builds() const {
    return 0;
  }

  virtual ~KernelLauncher() = default;
};

}  // namespace quadrants::lang
