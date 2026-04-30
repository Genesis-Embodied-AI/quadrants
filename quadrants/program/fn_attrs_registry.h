#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace quadrants::lang {

// Central registry of LLVM function attributes that users may set per-kernel
// via `@qd.kernel(fn_attrs={...})`. Keyed by backend short name (matching
// the namespacing used in the Python dict, e.g. "amdgpu"). Adding a new
// attribute means: (1) register it here, (2) ensure the consuming backend
// honors it (for AMDGPU: codegen_llvm.cpp applies it via addFnAttr; if the
// backend has a hardcoded default for the same key, gate that default with
// `!F.hasFnAttribute(key)` in the corresponding jit_*.cpp).
inline const std::unordered_map<std::string, std::unordered_set<std::string>> &
get_fn_attrs_registry() {
  static const std::unordered_map<std::string,
                                  std::unordered_set<std::string>>
      kRegistry = {
          {"amdgpu",
           {
               "amdgpu-max-num-workgroups",
               "amdgpu-agpr-alloc",
               "amdgpu-waves-per-eu",
               "amdgpu-flat-work-group-size",
               "amdgpu-sched-strategy",
           }},
      };
  return kRegistry;
}

}  // namespace quadrants::lang
