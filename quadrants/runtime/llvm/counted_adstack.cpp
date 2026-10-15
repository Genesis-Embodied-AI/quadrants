#include "quadrants/runtime/llvm/llvm_runtime_executor.h"
#include "quadrants/program/launch_context_builder.h"
#define QD_RUNTIME_HOST
#include "quadrants/program/context.h"
#undef QD_RUNTIME_HOST
#include "quadrants/rhi/llvm/llvm_device.h"
#include "quadrants/rhi/cuda/cuda_driver.h"
#include <limits>

namespace quadrants::lang {

void LlvmRuntimeExecutor::prepare_counted_adstack(const AdStackSizingInfo &layout,
                                                 std::size_t workers,
                                                 LaunchContextBuilder &ctx) {
  if (!layout.per_thread_stride || !workers)
    return;
  const auto capacity = ctx.get_context().adstack_capacity;
  QD_ERROR_IF(!capacity, "Stack-based backward requires a counted forward inside qd.adstack");
  const auto limit = std::numeric_limits<std::size_t>::max();
  QD_ERROR_IF(capacity > limit / layout.per_thread_stride ||
                  workers > limit / (capacity * layout.per_thread_stride),
              "Measured adstack allocation size overflow");
  const auto bytes = capacity * layout.per_thread_stride * workers;
  if (bytes <= adstack_heap_bytes_)
    return;
  synchronize();
  Device::AllocParams params{};
  params.size = bytes;
  params.usage = AllocUsage::Storage;
  DeviceAllocation allocation;
  QD_ERROR_IF(llvm_device()->allocate_memory(params, &allocation) != RhiResult::success,
              "Failed to allocate {} bytes for measured adstack", bytes);
  auto guard = std::make_unique<DeviceAllocationGuard>(std::move(allocation));
  void *ptr = get_device_alloc_info_ptr(*guard);
  if (!adstack_heap_field_) {
    get_runtime_jit_module()->call<void *>("runtime_get_adstack_heap_field_ptr", llvm_runtime_);
    adstack_heap_field_ = reinterpret_cast<void *>(fetch_result_uint64(quadrants_result_buffer_ret_value_id,
                                                                     result_buffer_cache_));
  }
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().memcpy_host_to_device(adstack_heap_field_, &ptr, sizeof(ptr));
#endif
  } else {
    QD_ERROR_IF(!arch_is_cpu(config_.arch), "Measured adstack supports CPU and CUDA on LLVM backends");
    *static_cast<void **>(adstack_heap_field_) = ptr;
  }
  adstack_heap_ = std::move(guard);
  adstack_heap_bytes_ = bytes;
}

}  // namespace quadrants::lang
