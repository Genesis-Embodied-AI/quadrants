#pragma once

#include "quadrants/codegen/compiled_kernel_data.h"
#include "quadrants/program/launch_context_builder.h"

namespace quadrants::lang {

class KernelLauncher {
 public:
  using Handle = KernelLaunchHandle;

  virtual void launch_kernel(const CompiledKernelData &compiled_kernel_data,
                             LaunchContextBuilder &ctx) = 0;

  virtual ~KernelLauncher() = default;
};

}  // namespace quadrants::lang
