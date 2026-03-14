#pragma once

#include "quadrants/codegen/compiled_kernel_data.h"
#include "quadrants/program/launch_context_builder.h"

namespace quadrants::lang {

class KernelLauncher {
 public:
  using Handle = KernelLaunchHandle;

  virtual void launch_kernel(const CompiledKernelData &compiled_kernel_data,
                             LaunchContextBuilder &ctx) = 0;

  virtual std::size_t get_cuda_graph_cache_size() const {
    return 0;
  }

  virtual bool get_cuda_graph_cache_used_on_last_call() const {
    return false;
  }

  virtual std::size_t get_cuda_graph_num_nodes_on_last_call() const {
    return 0;
  }

  virtual ~KernelLauncher() = default;
};

}  // namespace quadrants::lang
