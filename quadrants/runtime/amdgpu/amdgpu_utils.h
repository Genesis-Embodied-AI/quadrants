#pragma once

namespace quadrants::lang {
namespace amdgpu {

// Returns true if `ptr` refers to memory whose HIP attribute reports `hipMemoryTypeDevice`. Used to gate graph-build /
// device-only fast paths that cannot HtoD-stage host-resident pointers.
bool on_amdgpu_device(void *ptr);

}  // namespace amdgpu
}  // namespace quadrants::lang
