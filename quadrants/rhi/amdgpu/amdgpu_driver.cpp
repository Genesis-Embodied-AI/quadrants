#include "quadrants/rhi/amdgpu/amdgpu_driver.h"

#include "quadrants/common/dynamic_loader.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/util/environ_config.h"

namespace quadrants {
namespace lang {

std::string get_amdgpu_error_message(uint32 err) {
  auto err_name_ptr = AMDGPUDriver::get_instance_without_context().get_error_name(err);
  auto err_string_ptr = AMDGPUDriver::get_instance_without_context().get_error_string(err);
  return fmt::format("AMDGPU Error {}: {}", err_name_ptr, err_string_ptr);
}

AMDGPUDriverBase::AMDGPUDriverBase() {
  disabled_by_env_ = (get_environ_config("QD_ENABLE_AMDGPU", 1) == 0);
  if (disabled_by_env_) {
    QD_TRACE("AMDGPU driver disabled by enviroment variable \"QD_ENABLE_AMDGPU\".");
  }
}

bool AMDGPUDriverBase::load_lib(std::string lib_linux) {
#if defined(QD_PLATFORM_LINUX)
  auto lib_name = lib_linux;
#else
  static_assert(false, "Quadrants AMDGPU driver supports only Linux.");
#endif

  loader_ = std::make_unique<DynamicLoader>(lib_name);
  if (!loader_->loaded()) {
    QD_WARN("{} lib not found.", lib_name);
    return false;
  } else {
    QD_TRACE("{} loaded!", lib_name);
    return true;
  }
}

bool AMDGPUDriver::detected() {
  return !disabled_by_env_ && loader_->loaded();
}

AMDGPUDriver::AMDGPUDriver() {
  if (!load_lib("libamdhip64.so"))
    return;

  loader_->load_function("hipGetErrorName", get_error_name);
  loader_->load_function("hipGetErrorString", get_error_string);
  loader_->load_function("hipDriverGetVersion", driver_get_version);
  loader_->load_function("hipRuntimeGetVersion", runtime_get_version);

  int version;
  driver_get_version(&version);
  QD_TRACE("AMDGPU driver API (v{}.{}) loaded.", version / 1000, version % 1000 / 10);

#define PER_AMDGPU_FUNCTION(name, symbol_name, ...) \
  name.set(loader_->load_function(#symbol_name));   \
  name.set_lock(&lock_);                            \
  name.set_names(#name, #symbol_name);
#include "quadrants/rhi/amdgpu/amdgpu_driver_functions.inc.h"
#undef PER_AMDGPU_FUNCTION
}

AMDGPUDriver &AMDGPUDriver::get_instance_without_context() {
  // Thread safety guaranteed by C++ compiler
  // Note this is never deleted until the process finishes
  static AMDGPUDriver *instance = new AMDGPUDriver();
  return *instance;
}

AMDGPUDriver &AMDGPUDriver::get_instance() {
  // initialize the AMDGPU context so that the driver APIs can be called later
  AMDGPUContext::get_instance();
  return get_instance_without_context();
}

void amdgpu_memset(void *ptr, uint8 value, std::size_t size) {
  // Prefer the 32-bit-pattern path (hipMemsetD32) for the aligned bulk, which
  // matches how the CUDA backend issues zero-init (cuMemsetD32_v2); the ROCm
  // byte-fill path (hipMemset) returns hipErrorInvalidValue for sizes above
  // 1 GiB on RDNA3 + ROCm 7, while the 32-bit-pattern path is unaffected.
  auto &driver = AMDGPUDriver::get_instance();
  auto *bytes = static_cast<uint8 *>(ptr);
  std::size_t aligned_bytes = size & ~std::size_t{3};
  if (aligned_bytes > 0) {
    uint32 pattern = static_cast<uint32>(value) * 0x01010101u;
    driver.memsetd32(bytes, pattern, aligned_bytes / 4);
  }
  // Residual 0-3 trailing bytes fall back to byte hipMemset, which is always
  // well below the size at which the ROCm issue manifests.
  if (aligned_bytes < size) {
    driver.memset(bytes + aligned_bytes, value, size - aligned_bytes);
  }
}

}  // namespace lang
}  // namespace quadrants
