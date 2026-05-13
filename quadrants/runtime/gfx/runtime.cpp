#include "quadrants/runtime/gfx/runtime.h"

#include <cstdio>
#include <cstdlib>

#include "quadrants/codegen/spirv/adstack_sizer_shader.h"
#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/program.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/common/exceptions.h"
#include "quadrants/common/filesystem.hpp"

#include <cstring>

// FIXME: (penguinliong) Special offer for `run_codegen`. Find a new home for it
// in the future.
#include "quadrants/codegen/spirv/spirv_codegen.h"

#include <chrono>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define QD_RUNTIME_HOST
#include "quadrants/program/context.h"
#undef QD_RUNTIME_HOST

namespace quadrants::lang {
namespace gfx {

namespace {

class HostDeviceContextBlitter {
 public:
  HostDeviceContextBlitter(const KernelContextAttributes *ctx_attribs,
                           LaunchContextBuilder &host_ctx,
                           Device *device,
                           DeviceAllocation *device_args_buffer,
                           DeviceAllocation *device_ret_buffer)
      : ctx_attribs_(ctx_attribs),
        host_ctx_(host_ctx),
        device_args_buffer_(device_args_buffer),
        device_ret_buffer_(device_ret_buffer),
        device_(device) {
  }

  void host_to_device(const std::unordered_map<int, DeviceAllocation> &ext_arrays,
                      const std::unordered_map<int, DeviceAllocation> &ext_array_grads,
                      const std::unordered_map<int, size_t> &ext_arr_size) {
    if (!ctx_attribs_->has_args()) {
      return;
    }

    void *device_base{nullptr};
    QD_ASSERT(device_->map(*device_args_buffer_, &device_base) == RhiResult::success);

    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg_kv = ctx_attribs_->args()[i];
      const auto &indices = arg_kv.first;
      const auto &arg = arg_kv.second;
      if (arg.is_array) {
        QD_ASSERT(indices.size() == 1);
        int arg_id = indices[0];
        if (host_ctx_.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone &&
            ext_arr_size.at(arg_id)) {
          // Only need to blit ext arrs (host array)
          auto access_it = std::find_if(ctx_attribs_->arr_access.begin(), ctx_attribs_->arr_access.end(),
                                        [indices](const auto &pair) -> bool { return pair.first == indices; });
          QD_ASSERT(access_it != ctx_attribs_->arr_access.end());
          uint32_t access = uint32_t(access_it->second);
          if (access & uint32_t(irpass::ExternalPtrAccess::READ)) {
            DeviceAllocation buffer = ext_arrays.at(arg_id);
            void *device_arr_ptr{nullptr};
            // `QD_ERROR_IF` (not `QD_ASSERT`) so the failure message names what was being mapped; a bare
            // `QD_ASSERT(... == RhiResult::success)` would throw but only surface the condition string, leaving
            // the user to guess which map call broke. `QD_ASSERT` is also always-on (not release-gated), so this
            // is purely a message-quality choice.
            QD_ERROR_IF(device_->map(buffer, &device_arr_ptr) != RhiResult::success,
                        "Failed to map ext arr data buffer for host_to_device blit");
            ArgArrayPtrKey data_ptr_idx{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
            const void *host_ptr = host_ctx_.array_ptrs[data_ptr_idx];
            std::memcpy(device_arr_ptr, host_ptr, ext_arr_size.at(arg_id));
            device_->unmap(buffer);
          }
          // Mirror the host gradient buffer into the per-arg device allocation so the kernel can read/accumulate
          // into it. Reverse-mode AD issues an atomic-like read-modify-write via the grad_ptr slot, which on
          // Metal/Vulkan can only target device memory; dereferencing the host pointer directly silently writes
          // to unrelated memory and leaves host-side gradients at zero.
          //
          // The blit is gated on the grad-slot access bits computed by
          // `irpass::detect_external_ptr_grad_access_in_task` and published per-arg in
          // `ctx_attribs_->grad_arr_access`. We mirror whenever any task in the kernel either reads or writes
          // the grad slot: READ covers atomic read-modify-writes (the pre-launch device state must match the
          // host), WRITE covers plain non-atomic partial stores (`x.grad[i] = val` on a torch/numpy tensor) -
          // without the mirror, the device buffer would retain allocator garbage at indices the kernel did not
          // touch and the symmetric d2h readback below would silently overwrite the user's host `.grad` with
          // that garbage. A kernel that never touches `.grad` (the typical forward pass of a reverse-mode
          // kernel) has both bits clear and skips the per-dispatch map + memcpy + unmap entirely.
          auto grad_access_it = std::find_if(ctx_attribs_->grad_arr_access.begin(), ctx_attribs_->grad_arr_access.end(),
                                             [indices](const auto &pair) -> bool { return pair.first == indices; });
          uint32_t grad_access =
              (grad_access_it != ctx_attribs_->grad_arr_access.end()) ? uint32_t(grad_access_it->second) : 0;
          constexpr uint32_t kGradReadWrite =
              uint32_t(irpass::ExternalPtrAccess::READ) | uint32_t(irpass::ExternalPtrAccess::WRITE);
          auto grad_it = ext_array_grads.find(arg_id);
          if (grad_it != ext_array_grads.end() && (grad_access & kGradReadWrite)) {
            DeviceAllocation grad_buffer = grad_it->second;
            void *device_grad_ptr{nullptr};
            QD_ERROR_IF(device_->map(grad_buffer, &device_grad_ptr) != RhiResult::success,
                        "Failed to map ext arr grad buffer for host_to_device blit");
            // `.at` (rather than operator[]) so we never default-insert a nullptr here; a missing grad_ptr_idx at
            // this point would be a bug (we only reach this branch when ext_array_grads already contains arg_id,
            // which in turn requires array_ptrs to carry a non-null entry for the same grad key).
            ArgArrayPtrKey grad_ptr_idx{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
            const void *host_grad_ptr = host_ctx_.array_ptrs.at(grad_ptr_idx);
            std::memcpy(device_grad_ptr, host_grad_ptr, ext_arr_size.at(arg_id));
            device_->unmap(grad_buffer);
          }
        }
        // Substitute in the device address.

        if ((host_ctx_.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone ||
             host_ctx_.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNdarray) &&
            device_->get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer)) {
          ArgArrayPtrKey grad_ptr_idx{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
          uint64_t addr = device_->get_memory_physical_pointer(ext_arrays.at(arg_id));
          auto grad_it = ext_array_grads.find(arg_id);
          uint64_t grad_addr = 0;
          if (grad_it != ext_array_grads.end()) {
            grad_addr = device_->get_memory_physical_pointer(grad_it->second);
          } else {
            auto host_grad_it = host_ctx_.array_ptrs.find(grad_ptr_idx);
            if (host_grad_it != host_ctx_.array_ptrs.end()) {
              grad_addr = (uint64_t)host_grad_it->second;
            }
          }
          host_ctx_.set_ndarray_ptrs(arg_id, addr, grad_addr);
        }
      }
    }

    std::memcpy(device_base, host_ctx_.get_context().arg_buffer, ctx_attribs_->args_bytes());

    device_->unmap(*device_args_buffer_);
  }

  bool device_to_host(CommandList *cmdlist,
                      const std::unordered_map<int, DeviceAllocation> &ext_arrays,
                      const std::unordered_map<int, DeviceAllocation> &ext_array_grads,
                      const std::unordered_map<int, size_t> &ext_arr_size) {
    if (ctx_attribs_->empty()) {
      return false;
    }

    bool require_sync = ctx_attribs_->rets().size() > 0;
    std::vector<DevicePtr> readback_dev_ptrs;
    std::vector<void *> readback_host_ptrs;
    std::vector<size_t> readback_sizes;

    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &kv = ctx_attribs_->args()[i];
      const auto &indices = kv.first;
      const auto &arg = kv.second;
      if (arg.is_array) {
        QD_ASSERT(indices.size() == 1);
        int arg_id = indices[0];
        if (host_ctx_.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone &&
            ext_arr_size.at(arg_id)) {
          auto access_it = std::find_if(ctx_attribs_->arr_access.begin(), ctx_attribs_->arr_access.end(),
                                        [indices](const auto &pair) -> bool { return pair.first == indices; });
          QD_ASSERT(access_it != ctx_attribs_->arr_access.end());
          uint32_t access = uint32_t(access_it->second);
          if (access & uint32_t(irpass::ExternalPtrAccess::WRITE)) {
            // Only need to blit ext arrs (host array)
            readback_dev_ptrs.push_back(ext_arrays.at(arg_id).get_ptr(0));
            readback_host_ptrs.push_back(host_ctx_.array_ptrs[{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY}]);
            readback_sizes.push_back(ext_arr_size.at(arg_id));
            require_sync = true;
            // Grad readback is gated on the grad-slot WRITE bit from `grad_arr_access`, mirroring the host_to_device
            // path's READ gate. A forward-only kernel with `arr_access.WRITE=1` but no grad touch would otherwise blit
            // an uninitialised device grad buffer back over the user's host `.grad`, silently corrupting initialised
            // gradients.
            auto grad_access_it =
                std::find_if(ctx_attribs_->grad_arr_access.begin(), ctx_attribs_->grad_arr_access.end(),
                             [indices](const auto &pair) -> bool { return pair.first == indices; });
            uint32_t grad_access =
                (grad_access_it != ctx_attribs_->grad_arr_access.end()) ? uint32_t(grad_access_it->second) : 0;
            auto grad_it = ext_array_grads.find(arg_id);
            if (grad_it != ext_array_grads.end() && (grad_access & uint32_t(irpass::ExternalPtrAccess::WRITE))) {
              readback_dev_ptrs.push_back(grad_it->second.get_ptr(0));
              // `.at` (rather than operator[]) so a missing grad_ptr_idx throws immediately instead of
              // default-inserting a nullptr that the readback below would treat as a destination address.
              // Matches the host_to_device path above.
              ArgArrayPtrKey grad_ptr_idx{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
              readback_host_ptrs.push_back(host_ctx_.array_ptrs.at(grad_ptr_idx));
              readback_sizes.push_back(ext_arr_size.at(arg_id));
              require_sync = true;
            }
          }
        }
      }
    }

    if (require_sync) {
      if (readback_sizes.size()) {
        // `readback_data` internally allocates a staging buffer, records a `buffer_copy` cmdlist, and submits it via
        // `submit_synced` (which on every backend drains the compute stream's pending cmdbufs). Both Metal's
        // `command_sync` and Vulkan's `vkQueueWaitIdle` drain everything queued, so the kernel cmdlist submitted just
        // above completes as part of the same wait the readback already pays. A separate `wait_idle()` here would
        // double the host-GPU round-trip per readback launch with no extra correctness; the `command_complete_sema`
        // dependency edge is what the readback's submit honors on Vulkan, and Metal ignores `wait_semaphores` but
        // drains in submission order so the kernel still completes before the staging buffer's `buffer_copy` runs.
        StreamSemaphore command_complete_sema = device_->get_compute_stream()->submit(cmdlist);
        QD_ASSERT(device_->readback_data(readback_dev_ptrs.data(), readback_host_ptrs.data(), readback_sizes.data(),
                                         int(readback_sizes.size()), {command_complete_sema}) == RhiResult::success);
      } else {
        device_->get_compute_stream()->submit_synced(cmdlist);
      }

      if (!ctx_attribs_->has_rets()) {
        return true;
      }
    } else {
      return false;
    }

    void *device_base{nullptr};
    QD_ASSERT(device_->map(*device_ret_buffer_, &device_base) == RhiResult::success);

    void *ctx_result_buffer = host_ctx_.get_context().result_buffer;
    std::memcpy(ctx_result_buffer, device_base, ctx_attribs_->rets_bytes());

    device_->unmap(*device_ret_buffer_);

    return true;
  }

  static std::unique_ptr<HostDeviceContextBlitter> maybe_make(const KernelContextAttributes *ctx_attribs,
                                                              LaunchContextBuilder &host_ctx,
                                                              Device *device,
                                                              DeviceAllocation *device_args_buffer,
                                                              DeviceAllocation *device_ret_buffer) {
    if (ctx_attribs->empty()) {
      return nullptr;
    }
    return std::make_unique<HostDeviceContextBlitter>(ctx_attribs, host_ctx, device, device_args_buffer,
                                                      device_ret_buffer);
  }

 private:
  const KernelContextAttributes *const ctx_attribs_;
  LaunchContextBuilder &host_ctx_;
  DeviceAllocation *const device_args_buffer_;
  DeviceAllocation *const device_ret_buffer_;
  Device *const device_;
};

}  // namespace

constexpr size_t kGtmpBufferSize = 1024 * 1024;
constexpr size_t kListGenBufferSize = 32 << 20;

// Info for launching a compiled Quadrants kernel, which consists of a series of
// Unified Device API pipelines.

CompiledQuadrantsKernel::CompiledQuadrantsKernel(const Params &ti_params)
    : ti_kernel_attribs_(*ti_params.ti_kernel_attribs), device_(ti_params.device) {
  input_buffers_[BufferType::GlobalTmps] = ti_params.global_tmps_buffer;
  input_buffers_[BufferType::ListGen] = ti_params.listgen_buffer;

  // Compiled_structs can be empty if loading a kernel from an AOT module as
  // the SNode are not re-compiled/structured. In this case, we assume a
  // single root buffer size configured from the AOT module.
  for (int root = 0; root < ti_params.num_snode_trees; ++root) {
    BufferInfo buffer = {BufferType::Root, root};
    input_buffers_[buffer] = ti_params.root_buffers[root];
  }

  const auto arg_sz = ti_kernel_attribs_.ctx_attribs.args_bytes();
  const auto ret_sz = ti_kernel_attribs_.ctx_attribs.rets_bytes();

  args_buffer_size_ = arg_sz;
  ret_buffer_size_ = ret_sz;

  const auto &task_attribs = ti_kernel_attribs_.tasks_attribs;
  const auto &spirv_bins = ti_params.spirv_bins;
  QD_ASSERT(task_attribs.size() == spirv_bins.size());

  for (int i = 0; i < task_attribs.size(); ++i) {
    PipelineSourceDesc source_desc{PipelineSourceType::spirv_binary, (void *)spirv_bins[i].data(),
                                   spirv_bins[i].size() * sizeof(uint32_t)};
    auto [vp, res] =
        ti_params.device->create_pipeline_unique(source_desc, task_attribs[i].name, ti_params.backend_cache);
    QD_ERROR_IF(res != RhiResult::success,
                "Failed to create pipeline for kernel task '{}' (RhiResult={}). The SPIR-V shader was rejected by the "
                "backend driver; see the preceding RHI log for the underlying diagnostic. On Metal, a common cause is "
                "exceeding Apple's MSL per-thread Function-scope footprint in reverse-mode AD kernels that use the "
                "adstack pipeline.",
                task_attribs[i].name, int(res));
    pipelines_.push_back(std::move(vp));
  }
}

const QuadrantsKernelAttributes &CompiledQuadrantsKernel::ti_kernel_attribs() const {
  return ti_kernel_attribs_;
}

size_t CompiledQuadrantsKernel::num_pipelines() const {
  return pipelines_.size();
}

size_t CompiledQuadrantsKernel::get_args_buffer_size() const {
  return args_buffer_size_;
}

size_t CompiledQuadrantsKernel::get_ret_buffer_size() const {
  return ret_buffer_size_;
}

Pipeline *CompiledQuadrantsKernel::get_pipeline(int i) {
  return pipelines_[i].get();
}

GfxRuntime::GfxRuntime(const Params &params)
    : device_(params.device), profiler_(params.profiler), program_impl_(params.program_impl) {
  current_cmdlist_pending_since_ = high_res_clock::now();
  init_nonroot_buffers();

  // Read pipeline cache from disk if available.
  std::filesystem::path cache_path(get_repo_dir());
  cache_path /= "rhi_cache.bin";
  std::vector<char> cache_data;
  if (std::filesystem::exists(cache_path)) {
    QD_TRACE("Loading pipeline cache from {}", cache_path.generic_string());
    std::ifstream cache_file(cache_path, std::ios::binary);
    cache_data.assign(std::istreambuf_iterator<char>(cache_file), std::istreambuf_iterator<char>());
  } else {
    QD_TRACE("Pipeline cache not found at {}", cache_path.generic_string());
  }
  auto [cache, res] = device_->create_pipeline_cache_unique(cache_data.size(), cache_data.data());
  if (res == RhiResult::success) {
    backend_cache_ = std::move(cache);
  }
}

GfxRuntime::~GfxRuntime() {
  // Set `finalizing_` before synchronize() so the adstack-overflow QD_ERROR_IF there short-circuits: a throw
  // from this implicitly-noexcept destructor would call std::terminate(). See the field's declaration comment.
  finalizing_ = true;
  synchronize();

  // Write pipeline cache back to disk.
  if (backend_cache_) {
    uint8_t *cache_data = (uint8_t *)backend_cache_->data();
    size_t cache_size = backend_cache_->size();
    if (cache_data) {
      std::filesystem::path cache_path = std::filesystem::path(get_repo_dir()) / "rhi_cache.bin";
      std::ofstream cache_file(cache_path, std::ios::binary | std::ios::trunc);
      std::ostreambuf_iterator<char> output_iterator(cache_file);
      std::copy(cache_data, cache_data + cache_size, output_iterator);
    }
    backend_cache_.reset();
  }

  {
    decltype(ti_kernels_) tmp;
    tmp.swap(ti_kernels_);
  }
  global_tmps_buffer_.reset();
  listgen_buffer_.reset();
}

GfxRuntime::KernelHandle GfxRuntime::register_quadrants_kernel(GfxRuntime::RegisterParams reg_params) {
  CompiledQuadrantsKernel::Params params;
  params.ti_kernel_attribs = &(reg_params.kernel_attribs);
  params.num_snode_trees = reg_params.num_snode_trees;
  params.device = device_;
  params.root_buffers = {};
  for (int root = 0; root < root_buffers_.size(); ++root) {
    params.root_buffers.push_back(root_buffers_[root].get());
  }
  params.global_tmps_buffer = global_tmps_buffer_.get();
  params.listgen_buffer = listgen_buffer_.get();
  params.backend_cache = backend_cache_.get();

  for (int i = 0; i < reg_params.task_spirv_source_codes.size(); ++i) {
    const auto &spirv_src = reg_params.task_spirv_source_codes[i];

    // If we can reach here, we have succeeded. Otherwise
    // std::optional::value() would have killed us.
    params.spirv_bins.push_back(std::move(spirv_src));
  }
  KernelHandle res;
  res.set_launch_id(ti_kernels_.size());
  ti_kernels_.push_back(std::make_unique<CompiledQuadrantsKernel>(params));
  return res;
}

void GfxRuntime::launch_kernel(KernelHandle handle, LaunchContextBuilder &host_ctx) {
  auto *ti_kernel = ti_kernels_[handle.get_launch_id()].get();

#if defined(__APPLE__)
  if (profiler_) {
    const int apple_max_query_pool_count = 32;
    int task_count = ti_kernel->ti_kernel_attribs().tasks_attribs.size();
    if (task_count > apple_max_query_pool_count) {
      QD_WARN(
          "Cannot concurrently profile more than 32 tasks in a single "
          "Quadrants "
          "kernel. Profiling aborted.");
      profiler_ = nullptr;
    } else if (device_->profiler_get_sampler_count() + task_count > apple_max_query_pool_count) {
      flush();
      device_->profiler_sync();
    }
  }
#endif

  std::unique_ptr<DeviceAllocationGuard> args_buffer{nullptr}, ret_buffer{nullptr};

  if (ti_kernel->get_args_buffer_size()) {
    // Needs both Uniform (the main kernel binds args as a uniform buffer) and Storage (the adstack sizer
    // pipeline binds the same buffer through a `rw_buffer` / storage_buffer descriptor to resolve ndarray
    // data pointers out of arg slots). Per VUID-VkDescriptorBufferInfo-buffer-02999, a buffer bound through
    // a storage_buffer descriptor must have been allocated with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`.
    auto [buf, res] =
        device_->allocate_memory_unique({ti_kernel->get_args_buffer_size(),
                                         /*host_write=*/true, /*host_read=*/false,
                                         /*export_sharing=*/false, AllocUsage::Uniform | AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate args buffer");
    args_buffer = std::move(buf);
  }

  if (ti_kernel->get_ret_buffer_size()) {
    auto [buf, res] = device_->allocate_memory_unique({ti_kernel->get_ret_buffer_size(),
                                                       /*host_write=*/false, /*host_read=*/true,
                                                       /*export_sharing=*/false, AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate ret buffer");
    ret_buffer = std::move(buf);
  }

  // Create context blitter
  auto ctx_blitter = HostDeviceContextBlitter::maybe_make(&ti_kernel->ti_kernel_attribs().ctx_attribs, host_ctx,
                                                          device_, args_buffer.get(), ret_buffer.get());

  // `any_arrays` contain both external arrays and NDArrays
  std::unordered_map<int, DeviceAllocation> any_arrays;
  // Side-allocated device buffers that mirror host gradient tensors, keyed by ndarray arg_id. Populated only for
  // ext arrays whose corresponding torch tensor has requires_grad=True.
  std::unordered_map<int, DeviceAllocation> ext_array_grads;
  // `ext_array_size` only holds the size of external arrays (host arrays)
  // As buffer size information is only needed when it needs to be allocated
  // and transferred by the host
  std::unordered_map<int, size_t> ext_array_size;

  // Prepare context buffers & arrays
  if (ctx_blitter) {
    QD_ASSERT(ti_kernel->get_args_buffer_size() || ti_kernel->get_ret_buffer_size());

    const auto &args = ti_kernel->ti_kernel_attribs().ctx_attribs.args();
    for (auto &kv : args) {
      const auto &indices = kv.first;
      const auto &arg = kv.second;
      if (arg.is_array) {
        QD_ASSERT(indices.size() == 1);
        int arg_id = indices[0];
        if (host_ctx.device_allocation_type[arg_id] != LaunchContextBuilder::DevAllocType::kNone) {
          DeviceAllocation devalloc = kDeviceNullAllocation;
          // NDArray
          const ArgArrayPtrKey key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
          if (host_ctx.array_ptrs.count(key)) {
            devalloc = *(DeviceAllocation *)(host_ctx.array_ptrs[key]);
          }

          if (host_ctx.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNdarray) {
            any_arrays[arg_id] = devalloc;
            ndarrays_in_use_.insert(devalloc.alloc_id);
            // Reverse-mode AD kernels bind the gradient ndarray through a separate StorageBuffer slot on
            // backends without physical_storage_buffer, so publish the grad device allocation alongside the
            // data one. Use `find` + non-null check rather than `count` + operator[]: earlier code paths on
            // the same LaunchContextBuilder may have read `(uint64)array_ptrs[grad_key]` via operator[],
            // which default-inserts a nullptr-valued entry if the key was missing. A subsequent `count` would
            // then return 1 and the downstream `*(DeviceAllocation *)` deref would segfault. Observed on
            // graph_do_while kernels whose LaunchContextBuilder is reused across iterations.
            const ArgArrayPtrKey grad_key{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
            auto grad_it = host_ctx.array_ptrs.find(grad_key);
            if (grad_it != host_ctx.array_ptrs.end() && grad_it->second != nullptr) {
              DeviceAllocation grad_devalloc = *(DeviceAllocation *)(grad_it->second);
              ext_array_grads[arg_id] = grad_devalloc;
              ndarrays_in_use_.insert(grad_devalloc.alloc_id);
            }
          } else {
            QD_NOT_IMPLEMENTED;
          }
        } else {
          ext_array_size[arg_id] = host_ctx.array_runtime_sizes[arg_id];
          auto arr_access = ti_kernel->ti_kernel_attribs().ctx_attribs.arr_access;
          auto access_it = std::find_if(arr_access.begin(), arr_access.end(),
                                        [indices](const auto &pair) -> bool { return pair.first == indices; });
          QD_ASSERT(access_it != arr_access.end());
          uint32_t access = uint32_t(access_it->second);
          // Alloc ext arr
          size_t alloc_size = std::max(size_t(32), ext_array_size.at(arg_id));
          bool host_write = access & uint32_t(irpass::ExternalPtrAccess::READ);
          auto [allocated, res] = device_->allocate_memory_unique(
              {alloc_size, host_write, false, /*export_sharing=*/false, AllocUsage::Storage});
          QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate ext arr buffer");
          any_arrays[arg_id] = *allocated.get();
          ctx_buffers_.push_back(std::move(allocated));
          // Allocate a parallel device buffer for the gradient slot whenever the caller supplied a grad tensor.
          // Reverse-mode AD reads and writes into it, so we need both host_write and host_read to round-trip
          // host torch grads. `find` + non-null (instead of `count`) for the reason documented on the kNdarray
          // path above.
          const ArgArrayPtrKey grad_key{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
          auto grad_it = host_ctx.array_ptrs.find(grad_key);
          if (grad_it != host_ctx.array_ptrs.end() && grad_it->second != nullptr) {
            auto [grad_alloc, grad_res] = device_->allocate_memory_unique(
                {alloc_size, /*host_write=*/true, /*host_read=*/true, /*export_sharing=*/false, AllocUsage::Storage});
            QD_ASSERT_INFO(grad_res == RhiResult::success, "Failed to allocate ext arr grad buffer");
            ext_array_grads[arg_id] = *grad_alloc.get();
            ctx_buffers_.push_back(std::move(grad_alloc));
          }
        }
      }
    }

    ctx_blitter->host_to_device(any_arrays, ext_array_grads, ext_array_size);
  }

  // Record commands
  const auto &task_attribs = ti_kernel->ti_kernel_attribs().tasks_attribs;

  // Adstack-cache invalidation bump - see `bump_writes_for_kernel_spirv` in `program/adstack_size_expr_eval.{h,cpp}`.
  if (program_impl_ != nullptr) {
    bump_writes_for_kernel_spirv(program_impl_->program, &host_ctx, task_attribs,
                                 ti_kernel->ti_kernel_attribs().ctx_attribs.arr_access);
  }

  // Max-reducer dispatch. Must precede `publish_adstack_metadata_spirv` so the per-spec substitution lands before the
  // sizer's tree walk. Gated on whether any task in this kernel has captured specs so forward-only and reverse-mode-
  // without-recognized-MaxOverRange kernels skip the call entirely and pay zero per-launch overhead. Mirrors the
  // `any_lazy_task` gate below on `dispatch_adstack_bound_reducers`; implementation lives in
  // `runtime/gfx/adstack_max_reducer_launch.cpp`.
  const bool any_max_reducer_task =
      std::any_of(task_attribs.begin(), task_attribs.end(),
                  [](const spirv::TaskAttributes &t) { return !t.ad_stack.max_reducer_specs.empty(); });
  quadrants::lang::MaxReducerResultMap max_reducer_results;
  if (any_max_reducer_task) {
    max_reducer_results = dispatch_max_reducers(host_ctx, args_buffer.get(), any_arrays, task_attribs,
                                                ti_kernel->ti_kernel_attribs().name);
  }

  // Device-side adstack SizeExpr evaluation: every task with adstack allocas has its per-alloca `max_size` /
  // `offset` metadata resolved by a dedicated compute shader (see `quadrants/runtime/gfx/adstack_sizer_launch.cpp`
  // for the full mechanism). The helper internally early-returns (after seeding the per-task vector with
  // compile-time strides) when no task has adstack allocas, so forward-only kernels pay only the cheap pre-populate
  // pass; the actual sizer dispatch + `wait_idle()` only fires for reverse-mode kernels.
  std::vector<PerTaskAdStackRuntime> per_task_ad_stack = publish_adstack_metadata_spirv(
      host_ctx, args_buffer.get(), any_arrays, task_attribs, ti_kernel->ti_kernel_attribs().name, max_reducer_results);

  // Static-IR-bound sparse-adstack-heap reducer dispatch. Gated on whether any task in this kernel has a captured
  // `bound_expr` - the codegen routes such tasks through the lazy LCA-block atomic-rmw row claim that reads
  // `AdStackBoundRowCapacity[task_id]`; without any such task the reducer would unconditionally `flush() +
  // wait_idle()` an empty stream just to early-return. Forward-only and reverse-mode-without-bound-expr kernels
  // therefore pay zero overhead here. Tasks with a captured `bound_expr` get a generic reducer compute shader
  // dispatch that counts gate-passing threads; the count sizes the float adstack heap allocation exactly in the
  // bind path below, instead of the dispatched-threads worst case.
  const bool any_lazy_task = std::any_of(task_attribs.begin(), task_attribs.end(), [](const spirv::TaskAttributes &t) {
    return t.ad_stack.bound_expr.has_value();
  });
  std::unordered_map<int, uint32_t> per_task_bound_count;
  if (any_lazy_task) {
    per_task_bound_count = dispatch_adstack_bound_reducers(host_ctx, args_buffer.get(), task_attribs);
  }

  ensure_current_cmdlist();

  for (int i = 0; i < task_attribs.size(); ++i) {
    const auto &attribs = task_attribs[i];
    auto vp = ti_kernel->get_pipeline(i);

    // Cap `advisory_total_num_threads` to the ACTUAL iteration count when the codegen was able to extract the range end
    // as a product of ndarray-shape lookups (see `RangeForAttributes::end_shape_product`). Without this cap, a grad
    // kernel whose range is runtime-determined (`const_end = false`) inherits `kMaxNumThreadsGridStrideLoop = 131072`
    // from the codegen fallback, and the adstack-heap sizing below multiplies that by the per-thread stride to request
    // (e.g.) 48 GB for a 1-iteration B=1 workload - exceeding Metal's `maxBufferLength` and producing a hard RHI error.
    // The in-shader grid-stride loop handles any dispatched thread count >= 1 correctly; a tight cap just means each
    // dispatched thread processes fewer strides of idle work.
    int effective_advisory_threads = attribs.advisory_total_num_threads;
    if (attribs.range_for_attribs && !attribs.range_for_attribs->end_shape_product.empty()) {
      const auto &range = *attribs.range_for_attribs;
      // `const_begin` is asserted true at codegen whenever `end_stmt` is populated (see the
      // `QD_ASSERT(stmt->const_begin)` in the `if (stmt->end_stmt)` branch of `spirv_codegen.cpp`), so `range.begin` is
      // the literal begin value, not a gtmp offset.
      int64_t iter_end = 1;
      for (const auto &ref : range.end_shape_product) {
        std::vector<int> indices = ref.arg_id;
        indices.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
        indices.push_back(ref.axis);
        iter_end *= int64_t(host_ctx.get_struct_arg<int32_t>(indices));
      }
      int64_t iter_count = std::max<int64_t>(0, iter_end - int64_t(range.begin));
      effective_advisory_threads =
          int(std::min<int64_t>(int64_t(effective_advisory_threads), std::max<int64_t>(1, iter_count)));
    }
    // Adstack-bearing tasks additionally cap at `kAdStackMaxConcurrentThreads`, matching the LLVM CUDA / AMDGPU
    // launchers' `kAdStackMaxConcurrentThreads = 65536` advisory cap. The per-thread int / float adstack heap rows
    // scale linearly with the dispatched thread count, so an uncapped 600k-thread MPM grid kernel would request
    // ~2.5 GB just for the int heap (`linear_thread_idx * stride_int_bytes`) on every reverse-mode launch - the same
    // kernel sizes to ~70 MB on LLVM thanks to that cap. SPIR-V's in-shader grid-stride loop handles the smaller
    // dispatch correctly: each launched invocation walks `i += grid_dim() * block_dim()` until it has covered the
    // full logical iteration count. Skip the cap on tasks without adstack allocas to keep forward-only and
    // adstack-free kernels at saturating throughput.
    constexpr int kAdStackMaxConcurrentThreads = 65536;
    if (!attribs.ad_stack.allocas.empty() && effective_advisory_threads > kAdStackMaxConcurrentThreads) {
      effective_advisory_threads = kAdStackMaxConcurrentThreads;
    }
    const int group_x = (effective_advisory_threads + attribs.advisory_num_threads_per_group - 1) /
                        attribs.advisory_num_threads_per_group;
    // Adstack metadata (runtime-evaluated stride and per-alloca `(offset, max_size)` u32 table) precomputed
    // before the cmdlist opened - see the `per_task_ad_stack` loop above. Zero-length `metadata` means the
    // task has no adstacks; `stride_float` / `stride_int` are still populated from the compile-time values
    // (both zero in the no-adstack case, a non-zero sum from the cache-hit fallback when allocas exist but
    // none of them captured a symbolic bound).
    const auto &ad_stack_metadata = per_task_ad_stack[i].metadata;
    const uint32_t ad_stack_stride_float = per_task_ad_stack[i].stride_float;
    const uint32_t ad_stack_stride_int = per_task_ad_stack[i].stride_int;

    std::unique_ptr<ShaderResourceSet> bindings = device_->create_resource_set_unique();
    for (auto &bind : attribs.buffer_binds) {
      // We might have to bind a invalid buffer (this is fine as long as
      // shader don't do anything with it)
      if (bind.buffer.type == BufferType::ExtArr) {
        const auto &src = bind.buffer.is_grad ? ext_array_grads : any_arrays;
        auto it = src.find(bind.buffer.root_id);
        bindings->rw_buffer(bind.binding, it != src.end() ? it->second : kDeviceNullAllocation);
      } else if (bind.buffer.type == BufferType::AdStackOverflow) {
        // Two-slot u32 buffer: [overflow_signal, task_registry_id]. The SPIR-V codegen task-end emit
        // writes a non-zero `stack_id + 1` into slot 0 via OpAtomicUMax whenever any push site overflowed
        // its `max_size`, and `cmpxchg(0, registry_id)` into slot 1 to record the offending
        // `AdStackSizingInfo`'s `Program::adstack_sizing_info_registry_` id (only the FIRST overflowing
        // task's id sticks; subsequent threads' cmpxchg fails harmlessly). Host `synchronize()` reads
        // both slots, raises with `Program::diagnose_adstack_overflow_message(task_id)`, and zeros both
        // for the next window.
        if (!adstack_overflow_buffer_) {
          auto [buf, res] =
              device_->allocate_memory_unique({2 * sizeof(uint32_t), /*host_write=*/true, /*host_read=*/true,
                                               /*export_sharing=*/false, AllocUsage::Storage});
          QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack overflow buffer");
          adstack_overflow_buffer_ = std::move(buf);
          current_cmdlist_->buffer_fill(adstack_overflow_buffer_->get_ptr(0), kBufferSizeEntireSize, /*data=*/0);
          current_cmdlist_->buffer_barrier(*adstack_overflow_buffer_);
        }
        bindings->rw_buffer(bind.binding, *adstack_overflow_buffer_);
      } else if (bind.buffer.type == BufferType::AdStackRowCounter) {
        // Per-task atomic-counter array (`uint[num_tasks_in_kernel]`) that the SPIR-V codegen `OpAtomicIAdd`s into at
        // the LCA-block claim site, slot `task_id_in_kernel`. Read back by the codegen-emitted defense-in-depth bounds
        // clamp at the same LCA-block - never by the host - so each task's claim count must persist across all tasks in
        // this kernel's task loop (i.e. across the inner `i in 0..task_attribs.size()` binds below). The buffer is
        // cleared exactly once per kernel-launch (gated on `i == 0`, the first task) so the next kernel-launch starts
        // from zero on every slot. Sized to fit `task_attribs.size()` slots and grown lazily on launches that exceed
        // the prior allocation.
        const size_t needed_size = std::max<size_t>(task_attribs.size(), 1) * sizeof(uint32_t);
        if (!adstack_row_counter_buffer_ || adstack_row_counter_buffer_size_ < needed_size) {
          auto [buf, res] = device_->allocate_memory_unique({needed_size, /*host_write=*/true, /*host_read=*/true,
                                                             /*export_sharing=*/false, AllocUsage::Storage});
          QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack row counter buffer (needed_size={})",
                         needed_size);
          adstack_row_counter_buffer_ = std::move(buf);
          adstack_row_counter_buffer_size_ = needed_size;
        }
        if (i == 0) {
          // First task of this kernel-launch: zero every slot so every per-task atomic counter starts at 0. Subsequent
          // task binds in the same launch leave the buffer alone - this task's claim count must not be clobbered by a
          // later task's bind, and the per-slot indexing in the codegen guarantees no cross-task collision.
          current_cmdlist_->buffer_fill(adstack_row_counter_buffer_->get_ptr(0), kBufferSizeEntireSize, /*data=*/0);
          current_cmdlist_->buffer_barrier(*adstack_row_counter_buffer_);
        }
        bindings->rw_buffer(bind.binding, *adstack_row_counter_buffer_);
      } else if (bind.buffer.type == BufferType::AdStackBoundRowCapacity) {
        // Per-task row capacity array populated by `dispatch_adstack_bound_reducers` before the main task bind loop
        // opens (slot `ti` carries the reducer count for tasks with a captured `bound_expr`, UINT32_MAX otherwise). The
        // codegen-emitted defense-in-depth bounds check at the float Lowest Common Ancestor (LCA) block reads this slot
        // to detect a reducer / main divergence and signal UINT32_MAX into AdStackOverflow on mismatch; bindings here
        // just route the existing buffer onto the descriptor without clearing or growing (those happen in the reducer
        // launcher). Forward-only kernels never see an `AdStackBoundRowCapacity` binding because no float adstack push
        // exists; defensive null bind keeps the RHI happy if the codegen ever requests this buffer without the launcher
        // having populated it.
        if (adstack_bound_row_capacity_buffer_) {
          bindings->rw_buffer(bind.binding, *adstack_bound_row_capacity_buffer_);
        } else {
          bindings->rw_buffer(bind.binding, kDeviceNullAllocation);
        }
      } else if (bind.buffer.type == BufferType::AdStackTaskRegistryId) {
        // Per-task `Program::adstack_sizing_info_registry_` ids written by the SPIR-V launcher in
        // `publish_adstack_metadata_spirv` immediately after registering each adstack-bearing task; slot
        // `ti` holds the registry id for that task (0 for tasks without adstacks). The codegen-emitted
        // task-end overflow check reads slot `task_id_in_kernel_` and `OpAtomicCompareExchange`'s the
        // value into `AdStackOverflow[1]` when the latter is still 0 - recording the FIRST overflowing
        // task's identity. The codegen-side gate on `task_has_adstack_push_` ensures forward-only tasks
        // never request this binding, so the buffer is always allocated by `publish_adstack_metadata_
        // spirv` whenever it is requested. The defensive null bind keeps the assertion path intact if
        // a future codegen path requests the binding without the launcher having allocated.
        if (adstack_task_registry_id_buffer_) {
          bindings->rw_buffer(bind.binding, *adstack_task_registry_id_buffer_);
        } else {
          bindings->rw_buffer(bind.binding, kDeviceNullAllocation);
        }
      } else if (bind.buffer.type == BufferType::AdStackHeapFloat) {
        // SPIR-V adstack primal/adjoint storage for f32 adstacks. Sized for `effective_rows`: the count of threads the
        // static-IR-bound reducer pre-counted as passing the captured gate, when the task has a captured `bound_expr`
        // consumable by the reducer; otherwise the dispatched-threads worst case (which is `group_x * block_dim`, the
        // advisory rounded up to a workgroup multiple, so threads past the advisory -which still own an `invoc_id *
        // stride` slice on the eager fallback path - stay in-bounds even if they ever reach a push). Grown on demand
        // and reused across launches; contents do not need to persist across kernels. On empty rows (`effective_rows ==
        // 0`) no push/pop can execute, so bind a null allocation instead of asking the RHI for a zero-sized buffer
        // (which trips `RHI_ASSERT(params.size > 0)` on Vulkan and fails similarly on Metal). The stride used here is
        // the per-launch value produced by `evaluate_adstack_size_expr` over every alloca (stored in
        // `ad_stack_stride_float`), not the compile-time `attribs.ad_stack.per_thread_stride_float_compile_time`.
        size_t dispatched_threads = size_t(group_x) * size_t(attribs.advisory_num_threads_per_group);
        size_t effective_rows = dispatched_threads;
        auto bound_count_it = per_task_bound_count.find(i);
        if (bound_count_it != per_task_bound_count.end()) {
          effective_rows = bound_count_it->second;
          // Clip by the captured loop trip count. Each loop iteration claims at most one row at the LCA-block
          // (one `atomic_add` per gating iteration), so the heap needs at most `trip_count` rows regardless of
          // how many cells of the gating SNode the reducer counted; without this, an oversized SNode (1024-cell
          // `selector` paired with a 64-iter loop) inflates the float heap to the SNode's cell count even
          // though the loop can only claim 64 rows. Both trip-count sources are gated by the
          // `dispatched_threads` ceiling so a `dynamic_gpu_range_for` that exceeds the SPIR-V dispatch cap and
          // serialises iterations across threads (each thread reaches the LCA-block multiple times) does not
          // accidentally undersize the heap. See `clip_effective_rows_by_loop_trip_count` in
          // `program/adstack_size_expr_eval.cpp` for the shared implementation (LLVM mirrors the same call).
          if (attribs.ad_stack.bound_expr.has_value()) {
            clip_effective_rows_by_loop_trip_count(effective_rows, *attribs.ad_stack.bound_expr, dispatched_threads,
                                                   program_impl_ != nullptr ? program_impl_->program : nullptr,
                                                   &host_ctx);
          }
        } else if (attribs.ad_stack.bound_expr.has_value()) {
          // Reaching here means the bound reducer skipped this `bound_expr`-captured task and `per_task_bound_count`
          // has no entry for slot `i`. The reducer's skip paths in `dispatch_adstack_bound_reducers` are: PSB
          // capability missing, Int64 capability missing, or the per-task f64-on-no-f64 filter at
          // `adstack_bound_reducer_launch.cpp:165-170` dropping an f64-captured gate on a device without
          // `spirv_has_float64`. Continuing past this point with a heuristic heap size (`ceil(last_observed * 1.5)`,
          // possibly capped at `dispatched_threads` or at `lazy_claim_iter_count_upper_bound`) leaves a
          // workload-uplift OOB hole: any launch whose actual LCA-block claim count exceeds the heuristic silently
          // writes past the heap end, and the divergence overflow signal in `spirv_codegen.cpp`'s LCA-block claim
          // emission cannot help (it reads the inert UINT32_MAX-default capacity slot, never trips). Hard-error here
          // instead - every backend Quadrants targets advertises PSB, Int64, and Float64 today, so reaching this
          // branch on a real device is either an internal-consistency bug in the reducer's filter or running on a
          // hypothetical legacy device that this code does not support. The diagnostic prints which cap is missing
          // so the failure mode is unambiguous.
          QD_ASSERT_INFO(device_->get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer),
                         "adstack heap-bind tertiary fallback for task '{}' on a device without "
                         "spirv_has_physical_storage_buffer: the static-bound reducer skipped its dispatch and there "
                         "is no safe heap-sizing path on this device. Adstack-bearing reverse-mode kernels require "
                         "PSB, Int64, and (for f64-captured gates) Float64; this device is not supported.",
                         attribs.name);
          QD_ASSERT_INFO(device_->get_caps().get(DeviceCapability::spirv_has_int64),
                         "adstack heap-bind tertiary fallback for task '{}' on a device without spirv_has_int64: "
                         "the static-bound reducer skipped its dispatch and there is no safe heap-sizing path on "
                         "this device. Adstack-bearing reverse-mode kernels require PSB, Int64, and (for "
                         "f64-captured gates) Float64; this device is not supported.",
                         attribs.name);
          // f64 gate captured but the device lacks `spirv_has_float64` - the per-task filter at
          // `adstack_bound_reducer_launch.cpp:165-170` drops these so the reducer never publishes a count, and
          // there is no safe heap-sizing path. Codegen at `spirv_ir_builder.cpp` hard-errors when emitting an
          // f64 type without the cap, so a kernel reaching this point on a no-f64 device implies an
          // internal-consistency bug in the codegen/cap negotiation; surface it cleanly.
          if (attribs.ad_stack.bound_expr->field_dtype_is_float && attribs.ad_stack.bound_expr->field_dtype_is_double) {
            QD_ASSERT_INFO(device_->get_caps().get(DeviceCapability::spirv_has_float64),
                           "adstack heap-bind tertiary fallback for task '{}' with an f64-captured gate on a "
                           "device without spirv_has_float64: the static-bound reducer filtered out the f64 arm "
                           "and there is no safe heap-sizing path. Adstack-bearing reverse-mode kernels with f64 "
                           "gates require Float64; this device is not supported.",
                           attribs.name);
          }
          QD_ERROR(
              "adstack heap-bind tertiary fallback fired for task '{}' on a device that has PSB, Int64, and Float64. "
              "The bound reducer should have matched this task; reaching here is an internal-consistency bug. File "
              "an issue with `QD_DUMP_IR=1 ...` output attached.",
              attribs.name);
        }
        // The shader uses u64 index arithmetic for `row_id * stride + offset + count` when the device has Int64;
        // without Int64 the shader falls back to u32 OpIMul, which silently wraps past 2^32 and aliases threads into
        // one another's heap slice. Assert at launch time rather than emit silent corruption. `effective_rows` is the
        // upper bound on the row index the kernel will produce (because the lazy LCA-block atomic claim hands out row
        // ids in [0, count) where count is exactly the value the reducer published into this task's slot before this
        // dispatch starts).
        QD_ASSERT_INFO(device_->get_caps().get(DeviceCapability::spirv_has_int64) ||
                           size_t(ad_stack_stride_float) * effective_rows <= std::numeric_limits<uint32_t>::max(),
                       "adstack f32 heap offset would overflow u32 on a device without Int64: "
                       "stride={} effective_rows={}",
                       ad_stack_stride_float, effective_rows);
        // Floor `effective_rows` at 1 when the codegen emitted a float-heap binding (`ad_stack_stride_float > 0`): the
        // bound-expr reducer can legitimately count 0 threads passing the gate (e.g. on a workload that exercises a
        // kernel whose gate never matches in the current scene), but Metal RHI rejects a null `DeviceAllocation` bind
        // on a slot the descriptor set declares - and the codegen still emits the slot for every task with float
        // adstacks, so we cannot route this through `kDeviceNullAllocation`. Allocating one unused row is correct: with
        // `effective_rows == 0` no thread ever reaches the LCA-block claim, so the row stays idle and incurs only
        // `stride_float * 4` bytes (typically a few hundred). For tasks without a float heap binding (`stride_float ==
        // 0`), the codegen does not emit this branch and we never get here.
        const size_t effective_rows_floored = std::max<size_t>(effective_rows, ad_stack_stride_float > 0 ? 1 : 0);
        size_t required = size_t(ad_stack_stride_float) * effective_rows_floored * sizeof(float);
        // `QD_DEBUG_ADSTACK=1` opt-in diagnostic. One line per task per launch describing the float heap-bind sizing
        // decision: which sizing source fired (`reducer_count` for tasks with a captured `bound_expr` whose reducer
        // populated `per_task_bound_count`, `worst_case_dispatched` otherwise) and the resulting required bytes.
        // Persistent so memory regressions can be debugged without re-instrumenting.
        if (std::getenv("QD_DEBUG_ADSTACK")) {
          const char *src = "worst_case_dispatched";
          if (bound_count_it != per_task_bound_count.end()) {
            src = "reducer_count";
          }
          std::fprintf(stderr,
                       "[adstack_heap] task='%s' kind=F src=%s effective_rows=%zu stride=%u required_bytes=%zu "
                       "(%.2f MB)\n",
                       attribs.name.c_str(), src, effective_rows, ad_stack_stride_float, required,
                       double(required) / (1024.0 * 1024.0));
          std::fflush(stderr);
        }
        if (required == 0) {
          bindings->rw_buffer(bind.binding, kDeviceNullAllocation);
        } else {
          if (!adstack_heap_buffer_float_ || adstack_heap_buffer_float_size_ < required) {
            // Amortized doubling: mirrors `LlvmRuntimeExecutor::ensure_adstack_heap`. Without it, a sequence of
            // launches with monotonically increasing dispatch sizes (e.g. BFS / frontier expansion) between
            // `synchronize()` calls would reallocate on every launch and leave every displaced buffer sitting in
            // `ctx_buffers_` until the next sync, accumulating O(K^2 * N) bytes of live-but-unused GPU memory. Doubling
            // bounds the reallocations at O(log K) and the live memory at O(K * N).
            size_t new_size = std::max(required, 2 * adstack_heap_buffer_float_size_);
            auto [buf, res] = device_->allocate_memory_unique(
                {new_size, /*host_write=*/false, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
            // Fallback when the amortized-doubling size overshoots a device limit (e.g. Metal's `maxBufferLength`
            // capping `2 * old_size` even when `required` alone would fit): retry at exactly `required` bytes before
            // aborting the process. Trade-off is losing amortization on the retry path; still correct because the next
            // grow will reset amortization against the new, smaller base.
            if (res != RhiResult::success && new_size > required) {
              new_size = required;
              std::tie(buf, res) = device_->allocate_memory_unique(
                  {new_size, /*host_write=*/false, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
            }
            QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack heap float buffer (size={})",
                           new_size);
            // Defer the old buffer's free until the current cmdlist is submitted and synced: the previous launch
            // may still be in flight and referencing the old allocation, so freeing it synchronously here (via
            // `DeviceAllocationGuard`'s destructor, which runs on the `std::move` reassignment below) would
            // produce a GPU-side use-after-free. `ctx_buffers_` is cleared in `synchronize()` /
            // `device_to_host(...)` after the stream has drained, which is exactly the lifetime we need.
            if (adstack_heap_buffer_float_) {
              ctx_buffers_.push_back(std::move(adstack_heap_buffer_float_));
            }
            adstack_heap_buffer_float_ = std::move(buf);
            adstack_heap_buffer_float_size_ = new_size;
          }
          bindings->rw_buffer(bind.binding, *adstack_heap_buffer_float_);
        }
      } else if (bind.buffer.type == BufferType::AdStackHeapInt) {
        // SPIR-V adstack primal/adjoint storage for i32 and u1 adstacks. Same grow-on-demand policy and same
        // empty-dispatch guard as the float buffer above. Same u32-overflow guard as the float branch too. Uses
        // the per-launch `ad_stack_stride_int` (SizeExpr-evaluated) rather than the compile-time value.
        size_t dispatched_threads = size_t(group_x) * size_t(attribs.advisory_num_threads_per_group);
        QD_ASSERT_INFO(device_->get_caps().get(DeviceCapability::spirv_has_int64) ||
                           size_t(ad_stack_stride_int) * dispatched_threads <= std::numeric_limits<uint32_t>::max(),
                       "adstack i32/u1 heap offset would overflow u32 on a device without Int64: "
                       "stride={} dispatched_threads={}",
                       ad_stack_stride_int, dispatched_threads);
        size_t required = size_t(ad_stack_stride_int) * dispatched_threads * sizeof(int32_t);
        if (std::getenv("QD_DEBUG_ADSTACK")) {
          std::fprintf(stderr,
                       "[adstack_heap] task='%s' kind=I src=worst_case_dispatched dispatched_threads=%zu "
                       "stride=%u required_bytes=%zu (%.2f MB)\n",
                       attribs.name.c_str(), dispatched_threads, ad_stack_stride_int, required,
                       double(required) / (1024.0 * 1024.0));
          std::fflush(stderr);
        }
        if (required == 0) {
          bindings->rw_buffer(bind.binding, kDeviceNullAllocation);
        } else {
          if (!adstack_heap_buffer_int_ || adstack_heap_buffer_int_size_ < required) {
            // Same amortized-doubling rationale as the float heap above.
            size_t new_size = std::max(required, 2 * adstack_heap_buffer_int_size_);
            auto [buf, res] = device_->allocate_memory_unique(
                {new_size, /*host_write=*/false, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
            // Same amortized-doubling overshoot fallback as the float heap above (Metal `maxBufferLength`).
            if (res != RhiResult::success && new_size > required) {
              new_size = required;
              std::tie(buf, res) = device_->allocate_memory_unique(
                  {new_size, /*host_write=*/false, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
            }
            QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack heap int buffer (size={})", new_size);
            if (adstack_heap_buffer_int_) {
              ctx_buffers_.push_back(std::move(adstack_heap_buffer_int_));
            }
            adstack_heap_buffer_int_ = std::move(buf);
            adstack_heap_buffer_int_size_ = new_size;
          }
          bindings->rw_buffer(bind.binding, *adstack_heap_buffer_int_);
        }
      } else if (bind.buffer.type == BufferType::AdStackMetadata) {
        // Per-dispatch u32 buffer carrying `[stride_float, stride_int, (offset_i, max_size_i)*]`. Populated
        // from `ad_stack_metadata` above (which evaluated each alloca's `SizeExpr`). Empty `allocas` means
        // the codegen never emitted a `BufferType::AdStackMetadata` bind in the first place, so this branch
        // is reached only when we have something to upload. A fresh device allocation is used per task
        // rather than a shared grow-on-demand slot: the bindings descriptor captures the buffer handle at
        // cmdlist record time, but the shader reads the contents at cmdlist submit+execute time. A shared
        // slot that is host-memcpy'd in this record-loop iteration would be overwritten by the next
        // iteration's host memcpy (record is host-synchronous, execute is deferred), so at submit time
        // every task's dispatch reads the LAST task's metadata - sibling stacks with smaller `max_size`
        // then appear to shrink earlier stacks' capacities and trip the in-kernel `count < max_size` guard
        // at unexpectedly low counts. Retire the per-task buffer into `ctx_buffers_` so it stays alive
        // until the next sync drains the cmdlist.
        QD_ASSERT_INFO(!ad_stack_metadata.empty(),
                       "AdStackMetadata bind requested for a task that recorded no adstack allocas");
        const size_t required = ad_stack_metadata.size() * sizeof(uint32_t);
        auto [metadata_buf, res] = device_->allocate_memory_unique(
            {required, /*host_write=*/true, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
        QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack metadata buffer (size={})", required);
        void *mapped = nullptr;
        RhiResult map_res = device_->map_range(metadata_buf->get_ptr(0), required, &mapped);
        QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack metadata buffer for host upload (size={})",
                       required);
        std::memcpy(mapped, ad_stack_metadata.data(), required);
        device_->unmap(*metadata_buf);
        bindings->rw_buffer(bind.binding, *metadata_buf);
        ctx_buffers_.push_back(std::move(metadata_buf));
      } else if (bind.buffer.type == BufferType::Args) {
        bindings->buffer(bind.binding, args_buffer ? *args_buffer : kDeviceNullAllocation);
      } else if (bind.buffer.type == BufferType::Rets) {
        bindings->rw_buffer(bind.binding, ret_buffer ? *ret_buffer : kDeviceNullAllocation);
      } else {
        DeviceAllocation *alloc = ti_kernel->get_buffer_bind(bind.buffer);
        bindings->rw_buffer(bind.binding, alloc ? *alloc : kDeviceNullAllocation);
      }
    }

    if (attribs.task_type == OffloadedTaskType::listgen) {
      for (auto &bind : attribs.buffer_binds) {
        if (bind.buffer.type == BufferType::ListGen) {
          // FIXME: properlly support multiple list
          current_cmdlist_->buffer_fill(ti_kernel->get_buffer_bind(bind.buffer)->get_ptr(0), kBufferSizeEntireSize,
                                        /*data=*/0);
          current_cmdlist_->buffer_barrier(*ti_kernel->get_buffer_bind(bind.buffer));
        }
      }
    }

    current_cmdlist_->bind_pipeline(vp);
    RhiResult status = current_cmdlist_->bind_shader_resources(bindings.get());
    QD_ERROR_IF(status != RhiResult::success, "Resource binding error : RhiResult({})", status);

    if (device_->get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer)) {
      for (const auto &[arg_id, alloc] : any_arrays) {
        current_cmdlist_->track_physical_buffer(alloc);
      }
      for (const auto &[arg_id, alloc] : ext_array_grads) {
        current_cmdlist_->track_physical_buffer(alloc);
      }
    }

    if (profiler_) {
      current_cmdlist_->begin_profiler_scope(attribs.name);
    }

    status = current_cmdlist_->dispatch(group_x);

    if (profiler_) {
      current_cmdlist_->end_profiler_scope();
    }

    QD_ERROR_IF(status != RhiResult::success, "Dispatch error : RhiResult({})", status);
    current_cmdlist_->memory_barrier();
  }

  // Keep context buffers used in this dispatch
  if (ti_kernel->get_args_buffer_size()) {
    ctx_buffers_.push_back(std::move(args_buffer));
  }
  if (ti_kernel->get_ret_buffer_size()) {
    ctx_buffers_.push_back(std::move(ret_buffer));
  }

  // If we need to host sync, sync and remove in-flight references
  if (ctx_blitter) {
    if (ctx_blitter->device_to_host(current_cmdlist_.get(), any_arrays, ext_array_grads, ext_array_size)) {
      current_cmdlist_ = nullptr;
      ctx_buffers_.clear();
    }
  }

  submit_current_cmdlist_if_timeout();
}

void GfxRuntime::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
  ensure_current_cmdlist();
  current_cmdlist_->buffer_barrier(src);
  current_cmdlist_->buffer_copy(dst, src, size);
  current_cmdlist_->buffer_barrier(dst);
}

void GfxRuntime::synchronize() {
  flush();
  device_->wait_idle();
  // Profiler support
  if (profiler_) {
    device_->profiler_sync();
    auto sampled_records = device_->profiler_flush_sampled_time();
    for (auto &record : sampled_records) {
      profiler_->insert_record(record.first, record.second);
    }
  }
  ctx_buffers_.clear();
  ndarrays_in_use_.clear();
  pending_launches_since_sync_ = 0;
  // Async adstack-overflow report: every launch in this sync window that overflowed wrote a non-zero sentinel into the
  // shared flag buffer. Read it now, raise if any kernel overflowed, and zero it so the next sync window starts clean.
  // This mirrors the CUDA async-error pattern: the error surfaces on the next synchronize() rather than per launch. The
  // map() here must stay after the `wait_idle()` above; otherwise a future refactor could reorder and we would race
  // against pending GPU writes.
  if (adstack_overflow_buffer_ && !finalizing_) {
    uint32_t flag_val = 0;
    uint32_t task_id_val = 0;
    void *mapped = nullptr;
    QD_ASSERT(device_->map(*adstack_overflow_buffer_, &mapped) == RhiResult::success);
    auto *slots = reinterpret_cast<uint32_t *>(mapped);
    flag_val = slots[0];
    task_id_val = slots[1];
    if (flag_val != 0) {
      slots[0] = 0;
      slots[1] = 0;
    }
    device_->unmap(*adstack_overflow_buffer_);
    // UINT32_MAX is the dedicated sentinel the codegen-emitted defense-in-depth bounds check at the float Lowest
    // Common Ancestor (LCA) block writes via OpAtomicUMax when `claimed_row >= bound_row_capacity` for a captured
    // `bound_expr`. The bound is the exact reducer count (see `adstack_bound_reducer_launch.cpp`), so on a correct
    // codegen this branch is never taken; reaching it indicates the reducer's count diverged from the main pass's
    // actual LCA-block-reaching thread count - an internal-consistency bug, not a user-recoverable condition. Surface
    // a distinct actionable diagnostic so the failure is attributable to this exact mechanism rather than getting
    // confused with the per-stack `stack_id+1` overflow signal below (whose sentinel range tops out at `num_ad_stacks`
    // and cannot collide with UINT32_MAX in any realistic kernel).
    QD_ERROR_IF(flag_val == std::numeric_limits<uint32_t>::max(),
                "Internal: static-IR-bound sparse-adstack-heap reducer count diverged from main pass's actual "
                "LCA-block claim count. The bound is supposed to be exact by construction; reaching this signal "
                "means the reducer and the main pass observed different threads passing the captured gating "
                "predicate. File a bug with the kernel IR via `QD_DUMP_IR=1` and a minimal repro.");
    if (flag_val != 0) {
      Program *prog = (program_impl_ != nullptr) ? program_impl_->program : nullptr;
      std::string diagnostic;
      if (prog != nullptr) {
        auto diag = prog->adstack_cache().diagnose_adstack_overflow(task_id_val);
        diagnostic = std::move(diag.message);
        // See `LlvmRuntimeExecutor::check_adstack_overflow` for the rationale; only invalidate when the sizer rerun
        // confirmed a stale cache (DLPack-bypass) so a Quadrants pre-pass bug is not silently masked.
        if (diag.confirmed_invalid_cache) {
          prog->adstack_cache().invalidate_all_per_task();
        }
      }
      throw QuadrantsAssertionError(
          fmt::format("Adstack overflow: a reverse-mode autodiff kernel pushed more elements than the adstack "
                      "capacity allows. Raised at the next Quadrants Python entry rather than at the offending "
                      "kernel launch. Offending adstack index within the task: {}.\n{}",
                      flag_val - 1, diagnostic));
    }
  }
  fflush(stdout);
}

StreamSemaphore GfxRuntime::flush() {
  StreamSemaphore sema;
  if (current_cmdlist_) {
    sema = device_->get_compute_stream()->submit(current_cmdlist_.get());
    current_cmdlist_ = nullptr;
    // Do NOT clear ctx_buffers_ here: submit() returns as soon as the cmdlist is queued, not when the GPU has finished
    // executing. The deferred-free buffers in ctx_buffers_ (e.g. the old adstack heap buffer left over after a
    // grow-on-demand resize) may still be referenced by commands in flight. Only `synchronize()` clears the vector,
    // after `wait_idle()` has drained the stream.
  } else {
    auto [cmdlist, res] = device_->get_compute_stream()->new_command_list_unique();
    QD_ASSERT(res == RhiResult::success);
    cmdlist->memory_barrier();
    sema = device_->get_compute_stream()->submit(cmdlist.get());
  }
  return sema;
}

Device *GfxRuntime::get_ti_device() const {
  return device_;
}

void GfxRuntime::ensure_current_cmdlist() {
  // Create new command list if current one is nullptr
  if (!current_cmdlist_) {
    current_cmdlist_pending_since_ = high_res_clock::now();
    auto [cmdlist, res] = device_->get_compute_stream()->new_command_list_unique();
    QD_ASSERT(res == RhiResult::success);
    current_cmdlist_ = std::move(cmdlist);
  }
}

void GfxRuntime::submit_current_cmdlist_if_timeout() {
  // If we have accumulated some work but does not require sync and if the accumulated cmdlist has been pending for some
  // time launch the cmdlist to start processing. When the MTLCommandQueue is shared with another framework, the cmdbuf
  // must reach the queue before any external op is enqueued for Metal's same-queue FIFO scheduling to order them, so
  // commit every launch unconditionally instead of waiting for the staleness threshold.
  if (current_cmdlist_) {
    bool force_flush = program_impl_ != nullptr && program_impl_->config != nullptr &&
                       program_impl_->config->external_metal_command_queue != 0;
    constexpr uint64_t max_pending_time = 2000;  // 2000us = 2ms
    auto duration = high_res_clock::now() - current_cmdlist_pending_since_;
    if (force_flush || std::chrono::duration_cast<std::chrono::microseconds>(duration).count() > max_pending_time) {
      flush();
    }
  }
  // Safety valve against unbounded GPU-side tracking growth on tight kernel-launch loops without any
  // intervening Python-side observable (host readback, `to_numpy`, field get, ...). Normally every
  // Quadrants workload touches such an observable between launches and the implicit `synchronize()` those
  // paths trigger drains the queue. `VulkanStream::submit` pushes every submitted cmdbuffer into
  // `submitted_cmdbuffers_` with a fence; the vector is only cleared on `command_sync()` (i.e. `wait_idle`
  // -> `synchronize()`). Workloads that just push kernels and then read the final state at the end
  // (MPM88, iterative simulations) accumulate one deferred-free batch per flush and can reach hundreds of
  // live fences and cmdbuffers, at which point MoltenVK's encoder state tracker SIGSEGVs inside
  // `MVKCommandEncoder::encodeCommands` (the failure mode was a clean SIGSEGV on repeated launches of the
  // 3-task MPM88 substep kernel). Forcing a drain every `kMaxPendingLaunches` launches keeps the queue
  // bounded; the threshold is large enough that typical workloads (which already touch a host observable
  // every iteration) never reach it, so the periodic `wait_idle` does not become a measurable stall. A
  // non-blocking polling variant that checks individual fences via `vkGetFenceStatus` would retire sets as
  // they complete without blocking, but that requires an RHI public-surface change (`bool is_signaled()
  // const` on `StreamSemaphoreObject` and per-backend implementations) and the motivating workload only
  // needs a coarse-grained drain, not per-fence polling.
  constexpr size_t kMaxPendingLaunches = 32;
  pending_launches_since_sync_ += 1;
  if (pending_launches_since_sync_ > kMaxPendingLaunches) {
    synchronize();
  }
}

void GfxRuntime::init_nonroot_buffers() {
  {
    auto [buf, res] = device_->allocate_memory_unique({kGtmpBufferSize,
                                                       /*host_write=*/false, /*host_read=*/false,
                                                       /*export_sharing=*/false, AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "gtmp allocation failed");
    global_tmps_buffer_ = std::move(buf);
  }

  {
    auto [buf, res] = device_->allocate_memory_unique({kListGenBufferSize,
                                                       /*host_write=*/false, /*host_read=*/false,
                                                       /*export_sharing=*/false, AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "listgen allocation failed");
    listgen_buffer_ = std::move(buf);
  }

  // Need to zero fill the buffers, otherwise there could be NaN.
  Stream *stream = device_->get_compute_stream();
  auto [cmdlist, res] = device_->get_compute_stream()->new_command_list_unique();
  QD_ASSERT(res == RhiResult::success);

  cmdlist->buffer_fill(global_tmps_buffer_->get_ptr(0), kBufferSizeEntireSize,
                       /*data=*/0);
  cmdlist->buffer_fill(listgen_buffer_->get_ptr(0), kBufferSizeEntireSize,
                       /*data=*/0);
  stream->submit_synced(cmdlist.get());
}

void GfxRuntime::add_root_buffer(size_t root_buffer_size) {
  if (root_buffer_size == 0) {
    root_buffer_size = 4;  // there might be empty roots
  }
  auto [new_buffer, res_buffer] = device_->allocate_memory_unique({root_buffer_size,
                                                                   /*host_write=*/false, /*host_read=*/false,
                                                                   /*export_sharing=*/false, AllocUsage::Storage});
  QD_ASSERT_INFO(res_buffer == RhiResult::success, "Failed to allocate root buffer");
  Stream *stream = device_->get_compute_stream();
  auto [cmdlist, res_cmdlist] = device_->get_compute_stream()->new_command_list_unique();
  QD_ASSERT(res_cmdlist == RhiResult::success);
  cmdlist->buffer_fill(new_buffer->get_ptr(0), kBufferSizeEntireSize,
                       /*data=*/0);
  stream->submit_synced(cmdlist.get());
  root_buffers_.push_back(std::move(new_buffer));
  // cache the root buffer size
  root_buffers_size_map_[root_buffers_.back().get()] = root_buffer_size;
}

DeviceAllocation *GfxRuntime::get_root_buffer(int id) const {
  if (id >= root_buffers_.size()) {
    QD_ERROR("root buffer id {} not found", id);
  }
  return root_buffers_[id].get();
}

size_t GfxRuntime::get_root_buffer_size(int id) const {
  auto it = root_buffers_size_map_.find(root_buffers_[id].get());
  if (id >= root_buffers_.size() || it == root_buffers_size_map_.end()) {
    QD_ERROR("root buffer id {} not found", id);
  }
  return it->second;
}

void GfxRuntime::enqueue_compute_op_lambda(std::function<void(Device *device, CommandList *cmdlist)> op,
                                           const std::vector<ComputeOpImageRef> &image_refs) {
  for (const auto &ref : image_refs) {
    QD_ASSERT(last_image_layouts_.find(ref.image.alloc_id) != last_image_layouts_.end());
  }

  ensure_current_cmdlist();
  op(device_, current_cmdlist_.get());

  for (const auto &ref : image_refs) {
    last_image_layouts_[ref.image.alloc_id] = ref.final_layout;
  }
}

GfxRuntime::RegisterParams run_codegen(Kernel *kernel,
                                       Arch arch,
                                       const DeviceCapabilityConfig &caps,
                                       const std::vector<CompiledSNodeStructs> &compiled_structs,
                                       const CompileConfig &compile_config) {
  const auto id = Program::get_kernel_id();
  const auto quadrants_kernel_name(fmt::format("{}_k{:04d}_vk", kernel->name, id));
  QD_TRACE("VK codegen for Quadrants kernel={}", quadrants_kernel_name);
  spirv::KernelCodegen::Params params;
  params.ti_kernel_name = quadrants_kernel_name;
  params.kernel = kernel;
  params.ir_root = kernel->ir.get();
  params.compiled_structs = compiled_structs;
  params.arch = arch;
  params.caps = caps;
  params.enable_spv_opt = compile_config.external_optimization_level > 0;
  params.compile_config = &compile_config;
  spirv::KernelCodegen codegen(params);
  GfxRuntime::RegisterParams res;
  codegen.run(res.kernel_attribs, res.task_spirv_source_codes);
  res.num_snode_trees = compiled_structs.size();
  return res;
}

std::pair<const lang::StructType *, size_t> GfxRuntime::get_struct_type_with_data_layout(const lang::StructType *old_ty,
                                                                                         const std::string &layout) {
  auto [new_ty, size, align] = get_struct_type_with_data_layout_impl(old_ty, layout);
  return {new_ty, size};
}

std::tuple<const lang::StructType *, size_t, size_t> GfxRuntime::get_struct_type_with_data_layout_impl(
    const lang::StructType *old_ty,
    const std::string &layout) {
  QD_TRACE("get_struct_type_with_data_layout: {}", layout);
  QD_ASSERT(layout.size() == 2);
  auto is_430 = layout[0] == '4';
  auto has_buffer_ptr = layout[1] == 'b';
  auto members = old_ty->elements();
  size_t bytes = 0;
  size_t align = 0;
  for (int i = 0; i < members.size(); i++) {
    auto &member = members[i];
    size_t member_align;
    size_t member_size;
    if (auto struct_type = member.type->cast<lang::StructType>()) {
      auto [new_ty, size, member_align_] = get_struct_type_with_data_layout_impl(struct_type, layout);
      members[i].type = new_ty;
      member_align = member_align_;
      member_size = size;
    } else if (auto tensor_type = member.type->cast<lang::TensorType>()) {
      size_t element_size = data_type_size_gfx(tensor_type->get_element_type());
      size_t num_elements = tensor_type->get_num_elements();
      if (!is_430) {
        if (num_elements == 2) {
          member_align = element_size * 2;
        } else {
          member_align = element_size * 4;
        }
        member_size = member_align;
      } else {
        member_align = element_size;
        member_size = tensor_type->get_num_elements() * element_size;
      }
    } else if (auto pointer_type = member.type->cast<PointerType>()) {
      if (has_buffer_ptr) {
        member_size = sizeof(uint64_t);
        member_align = member_size;
      } else {
        // Use u32 as placeholder
        member_size = sizeof(uint32_t);
        member_align = member_size;
      }
    } else {
      QD_ASSERT(member.type->is<PrimitiveType>());
      member_size = data_type_size_gfx(member.type);
      member_align = member_size;
    }
    bytes = align_up(bytes, member_align);
    members[i].offset = bytes;
    bytes += member_size;
    align = std::max(align, member_align);
  }

  if (!is_430) {
    align = align_up(align, sizeof(float) * 4);
    bytes = align_up(bytes, 4 * sizeof(float));
  }
  QD_TRACE("  total_bytes={}", bytes);
  return {TypeFactory::get_instance().get_struct_type(members, layout)->as<lang::StructType>(), bytes, align};
}

}  // namespace gfx
}  // namespace quadrants::lang
