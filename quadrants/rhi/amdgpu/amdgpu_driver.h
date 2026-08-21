#pragma once

#include <exception>
#include <mutex>

#include "quadrants/common/dynamic_loader.h"

namespace quadrants {
namespace lang {

constexpr uint32 HIP_EVENT_DEFAULT = 0x0;
constexpr uint32 HIP_STREAM_DEFAULT = 0x0;
constexpr uint32 HIP_STREAM_NON_BLOCKING = 0x1;
constexpr uint32 HIP_MEM_ATTACH_GLOBAL = 0x1;
constexpr uint32 HIP_MEM_ADVISE_SET_PREFERRED_LOCATION = 3;
constexpr uint32 HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 26;
constexpr uint32 HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 63;
// hipDeviceAttributeMemoryPoolsSupported from the hipDeviceAttribute_t enum in
// ROCm/clr hipamd/include/hip/hip_runtime_api.h.
constexpr uint32 HIP_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 88;
// hipMemPoolAttrReleaseThreshold from the hipMemPoolAttr enum in
// ROCm/clr hipamd/include/hip/hip_runtime_api.h.
constexpr uint32 HIP_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4;
// sizeof(hipDeviceProperties_t) in ROCm 6.
// ROCm 5.7.1 is 792 and ROCm 6 is 1472, so to make both work we use whichever
// is larger.
constexpr uint32 HIP_DEVICE_PROPERTIES_STRUCT_SIZE = 1472;
// offsetof(hipDeviceProp_t, gcnArchName) / 4
constexpr uint32 HIP_DEVICE_GCN_ARCH_NAME = 396 / 4;
// offsetof(hipDeviceProp_t, gcnArchName) / 4
constexpr uint32 HIP_DEVICE_GCN_ARCH_NAME_6 = 1160 / 4;
// offsetof(hipDeviceProp_t, major) / 4
constexpr uint32 HIP_DEVICE_MAJOR = 328 / 4;
// offsetof(hipDeviceProp_t, major) / 4
constexpr uint32 HIP_DEVICE_MAJOR_6 = 360 / 4;
// offsetof(hipDeviceProp_t, minor) / 4
constexpr uint32 HIP_DEVICE_MINOR = 332 / 4;
// offsetof(hipDeviceProp_t, minor) / 4
constexpr uint32 HIP_DEVICE_MINOR_6 = 364 / 4;
constexpr uint32 HIP_ERROR_ASSERT = 710;
// hipErrorLaunchFailure — returned by the first hipStreamSynchronize after a device
// __builtin_trap() (AMDGPU in-kernel assert path). Catchable; does not abort the process.
constexpr uint32 HIP_ERROR_LAUNCH_FAILURE = 719;
constexpr uint32 HIP_JIT_MAX_REGISTERS = 0;
constexpr uint32 HIP_POINTER_ATTRIBUTE_MEMORY_TYPE = 2;
constexpr uint32 HIP_SUCCESS = 0;
constexpr uint32 HIP_MEMORYTYPE_DEVICE = 1;
// hipHostMallocCoherent — required for host-visible reads of assert state after a trap.
constexpr uint32 HIP_HOST_MALLOC_COHERENT = 0x40000000;

// Optional hook invoked from AMDGPUFunction::operator() on HIP_ERROR_LAUNCH_FAILURE before the
// generic QD_ERROR. LlvmRuntimeExecutor registers this in debug+amdgpu mode to surface
// QuadrantsAssertionError from pinned assert state. May throw.
using AmdgpuLaunchFailureHook = void (*)();
void set_amdgpu_launch_failure_hook(AmdgpuLaunchFailureHook hook);
AmdgpuLaunchFailureHook get_amdgpu_launch_failure_hook();
// True after a debug-mode device assert has been surfaced as QuadrantsAssertionError.
// The HIP context is dead afterward, so subsequent calls also return launch failure.
bool amdgpu_device_assert_already_surfaced();
void amdgpu_reset_device_assert_surfaced_flag();
void amdgpu_mark_device_assert_surfaced();
// True while a Program/executor is tearing down. Dead-context launch failures are only
// suppressed inside this window (or while unwinding the just-thrown assertion) so destructors
// do not std::terminate(); outside it, post-assert launch failures are surfaced as hard errors
// instead of silently reported as success on a dead context. Set by LlvmRuntimeExecutor.
bool amdgpu_device_in_teardown();
void amdgpu_set_device_in_teardown(bool in_teardown);
// `hipFuncAttributeMaxDynamicSharedMemorySize` from the `hipFuncAttribute` enum in ROCm/clr
// hipamd/include/hip/hip_runtime_api.h. Used with `kernel_set_attribute` (`hipFuncSetAttribute`) to opt in to >48 KB
// of dynamic shared memory for graph kernel nodes that request it.
constexpr uint32 HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;

std::string get_amdgpu_error_message(uint32 err);

template <typename... Args>
class AMDGPUFunction {
 public:
  AMDGPUFunction() {
    function_ = nullptr;
  }

  void set(void *func_ptr) {
    function_ = (func_type *)func_ptr;
  }

  uint32 call(Args... args) {
    QD_ASSERT(function_ != nullptr);
    QD_ASSERT(driver_lock_ != nullptr);
    std::lock_guard<std::mutex> _(*driver_lock_);
    return (uint32)function_(args...);
  }

  void set_names(const std::string &name, const std::string &symbol_name) {
    name_ = name;
    symbol_name_ = symbol_name;
  }

  void set_lock(std::mutex *lock) {
    driver_lock_ = lock;
  }

  std::string get_error_message(uint32 err) {
    return get_amdgpu_error_message(err) + fmt::format(" while calling {} ({})", name_, symbol_name_);
  }

  uint32 call_with_warning(Args... args) {
    auto err = call(args...);
    QD_WARN_IF(err, "{}", get_error_message(err));
    return err;
  }

  void operator()(Args... args) {
    auto err = call(args...);
    // Intercept launch failure before the generic HIP error so a debug-mode in-kernel assert
    // can raise QuadrantsAssertionError from pinned host memory (context is dead after trap).
    if (err == HIP_ERROR_LAUNCH_FAILURE) {
      if (auto hook = get_amdgpu_launch_failure_hook()) {
        hook();  // may throw QuadrantsAssertionError on first surfacing
      }
      if (amdgpu_device_assert_already_surfaced()) {
        // The HIP context is dead after the trap, so every later call also returns launch
        // failure. Suppress only where throwing would std::terminate(): inside teardown
        // destructors, or while unwinding the just-thrown QuadrantsAssertionError. Anywhere
        // else (e.g. user code that caught the assertion and kept issuing GPU work) fail
        // loudly instead of returning stale/uninitialized results as success.
        if (amdgpu_device_in_teardown() || std::uncaught_exceptions() > 0) {
          return;
        }
        QD_ERROR(
            "AMDGPU device context is unusable after an in-kernel assertion failure; "
            "re-initialize Quadrants in a fresh process before issuing further GPU work "
            "(while calling {} ({}))",
            name_, symbol_name_);
      }
    }
    QD_ERROR_IF(err, get_error_message(err));
  }

 private:
  using func_type = uint32_t(Args...);

  func_type *function_{nullptr};
  std::string name_, symbol_name_;
  std::mutex *driver_lock_{nullptr};
};

class AMDGPUDriverBase {
 public:
  ~AMDGPUDriverBase() = default;

 protected:
  std::unique_ptr<DynamicLoader> loader_;
  AMDGPUDriverBase();

  bool load_lib(std::string lib_linux);

  bool disabled_by_env_{false};
};

class AMDGPUDriver : protected AMDGPUDriverBase {
 public:
#define PER_AMDGPU_FUNCTION(name, symbol_name, ...) AMDGPUFunction<__VA_ARGS__> name;
#include "quadrants/rhi/amdgpu/amdgpu_driver_functions.inc.h"
#undef PER_AMDGPU_FUNCTION

  char *(*get_error_name)(uint32);

  char *(*get_error_string)(uint32);

  void (*driver_get_version)(int *);

  void (*runtime_get_version)(int *);

  bool detected();

  static AMDGPUDriver &get_instance();

  static AMDGPUDriver &get_instance_without_context();

  // Thin wrappers that transparently fall back to the synchronous hipMalloc / hipFree when the device does not
  // advertise memory-pool support. Mirrors CUDADriver::{malloc_async, mem_free_async}.
  void malloc_async(void **dev_ptr, size_t size, void *stream);
  void mem_free_async(void *dev_ptr, void *stream);

 private:
  AMDGPUDriver();

  std::mutex lock_;

  // bool rocm_version_valid_{false};
};

}  // namespace lang
}  // namespace quadrants
