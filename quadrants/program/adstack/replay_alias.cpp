#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program.h"

namespace quadrants::lang {

void LaunchContextBuilder::check_adstack_replay_aliases() {
  struct Array {
    uintptr_t address;
    uint64_t bytes;
    DeviceAllocation *allocation;
    bool needs_grad;
  };
  std::vector<Array> arrays;
  const bool llvm = arch_uses_llvm(kernel_->program->compile_config().arch);
  for (const auto &[id, parameter] : kernel_->nested_parameters) {
    if (!parameter.is_array)
      continue;
    auto ptr = array_ptrs.find({id, TypeFactory::DATA_PTR_POS_IN_NDARRAY});
    if (ptr == array_ptrs.end() || !ptr->second || !array_runtime_sizes.at(id))
      continue;
    auto *allocation = device_allocation_type.at(id) == DevAllocType::kNdarray
                           ? static_cast<DeviceAllocation *>(ptr->second)
                           : nullptr;
    uintptr_t address = reinterpret_cast<uintptr_t>(ptr->second);
    uint64_t bytes = array_runtime_sizes.at(id);
    if (allocation) {
      bytes *= parameter.get_element_size();
      if (llvm)
        address = reinterpret_cast<uintptr_t>(kernel_->program->get_program_impl()->get_device_alloc_info_ptr(*allocation));
    }
    arrays.push_back({address, bytes, allocation, parameter.needs_grad});
  }
  for (size_t i = 0; i < arrays.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      const auto &a = arrays[i];
      const auto &b = arrays[j];
      // Only non-differentiable output writes survive MakeAdjoint. Read-only aliases between differentiated
      // arguments need no extra restriction. Be conservative for non-differentiable arguments in this prototype.
      if (a.needs_grad && b.needs_grad)
        continue;
      bool overlaps;
      if (!llvm && (a.allocation || b.allocation)) {
        QD_ERROR_IF(!a.allocation || !b.allocation,
                    "Adstack replay counting cannot check mixed native/external array aliases on this backend");
        overlaps = *a.allocation == *b.allocation;
      } else {
        overlaps = a.address <= b.address ? b.address - a.address < a.bytes : a.address - b.address < b.bytes;
      }
      QD_ERROR_IF(overlaps, "Adstack replay counting does not support aliased non-differentiable array arguments");
    }
  }
}

}  // namespace quadrants::lang
