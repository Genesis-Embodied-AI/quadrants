#include "quadrants/runtime/gfx/runtime.h"
#include "quadrants/codegen/spirv/adstack_sizer_shader.h"
#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/program.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/ir/type_factory.h"
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
          // DO NOT gate this blit on `access & WRITE`. `access` is derived from the kernel's access analysis
          // over the *data* slot; it does not track read/write of the *grad* slot. A backward kernel that
          // reads `loss.grad[None]` as the reverse-mode seed (and writes `a.grad[i]`) has `access(loss) = READ`
          // only - WRITE is unset. Skipping the grad blit for that case leaves the device `loss.grad` stale
          // or zeroed, the backward's atomic read-modify-write seeds from zero, and every `a.grad[i]` comes
          // out zero. The unconditional blit has a measurable but bounded per-dispatch cost (one map+memcpy+unmap
          // per grad-bearing ndarray); a future correct optimisation would need a grad-specific access flag, not
          // the data-slot `access` here.
          auto grad_it = ext_array_grads.find(arg_id);
          if (grad_it != ext_array_grads.end()) {
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
            // Grad readback is gated on the same WRITE bit as the data readback because `arr_access` is
            // derived from the kernel's static access analysis and covers data+grad together. Forward-only
            // kernels have WRITE cleared, so skipping grad readback there avoids a GPU sync + DMA on every
            // forward dispatch once `.grad` buffers exist. Without this guard, a training loop's forward
            // pass would call `wait_idle()` + readback the (unchanged) grad buffer after the first backward
            // creates the grad allocations, roughly doubling forward latency on Metal/Vulkan.
            auto grad_it = ext_array_grads.find(arg_id);
            if (grad_it != ext_array_grads.end()) {
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
        StreamSemaphore command_complete_sema = device_->get_compute_stream()->submit(cmdlist);

        device_->wait_idle();

        // In this case `readback_data` syncs
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
    auto [buf, res] = device_->allocate_memory_unique({ti_kernel->get_args_buffer_size(),
                                                       /*host_write=*/true, /*host_read=*/false,
                                                       /*export_sharing=*/false, AllocUsage::Uniform});
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

  // Device-side adstack SizeExpr evaluation: every task with adstack allocas has its per-alloca `max_size` /
  // `offset` metadata resolved by a dedicated compute shader (`quadrants/codegen/spirv/adstack_sizer_shader`).
  // The shader reads the ndarray data pointer straight out of the kernel arg buffer via Physical Storage
  // Buffer addressing and dereferences where the memory lives, which is the only way to resolve an
  // `ExternalTensorRead` against a GPU-private `qd.ndarray` without round-tripping the entire ndarray
  // through host memory. Host-eval was the old path; it produced garbage for `kNdarray` args because the
  // host can't dereference GPU-private storage, which is why the on-device interpreter is mandatory rather
  // than optional on this backend.
  struct TaskAdStackRuntime {
    std::vector<uint32_t> metadata;  // [stride_float, stride_int, (offset, max_size)*]
    uint32_t stride_float{0};
    uint32_t stride_int{0};
  };
  std::vector<TaskAdStackRuntime> per_task_ad_stack(task_attribs.size());
  for (size_t ti = 0; ti < task_attribs.size(); ++ti) {
    per_task_ad_stack[ti].stride_float = task_attribs[ti].ad_stack.per_thread_stride_float_compile_time;
    per_task_ad_stack[ti].stride_int = task_attribs[ti].ad_stack.per_thread_stride_int_compile_time;
  }

  std::vector<size_t> adstack_task_indices;
  for (size_t ti = 0; ti < task_attribs.size(); ++ti) {
    if (!task_attribs[ti].ad_stack.allocas.empty())
      adstack_task_indices.push_back(ti);
  }

  if (!adstack_task_indices.empty()) {
    QD_ASSERT_INFO(program_impl_ != nullptr && program_impl_->program != nullptr,
                   "GfxRuntime::launch_kernel: `ProgramImpl::program` back-reference not set; cannot "
                   "encode AdStack SizeExpr bytecode. Ensure GfxProgramImpl passes `program_impl = this` "
                   "into `GfxRuntime::Params`.");
    QD_ERROR_IF(!device_->get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer) ||
                    !device_->get_caps().get(DeviceCapability::spirv_has_int64),
                "GfxRuntime::launch_kernel: the on-device adstack SizeExpr sizer requires both the Physical Storage "
                "Buffer and Int64 SPIR-V capabilities, but at least one is missing on this device. There is no "
                "correct host-eval fallback for `qd.ndarray`-backed reverse-mode state on a GPU-private backend; "
                "the shader must run on-device or the kernel's adstack sizing is garbage. Use a backend that "
                "advertises both caps (e.g. Metal on Apple Silicon, Vulkan 1.2+ with "
                "`VK_KHR_buffer_device_address`), or run the workload on the LLVM runtime (CPU / CUDA / AMDGPU).");

    // Build the sizer pipeline on first use.
    if (!adstack_sizer_pipeline_) {
      std::vector<uint32_t> spirv = spirv::build_adstack_sizer_spirv(Arch::vulkan, &device_->get_caps());
      QD_ASSERT_INFO(!spirv.empty(),
                     "`build_adstack_sizer_spirv` returned an empty binary despite the PSB+Int64 capability "
                     "check passing; bug in the shader builder's capability gating.");
      PipelineSourceDesc source_desc{PipelineSourceType::spirv_binary, (void *)spirv.data(),
                                     spirv.size() * sizeof(uint32_t)};
      auto [pipeline, res] = device_->create_pipeline_unique(source_desc, "adstack_sizer", backend_cache_.get());
      QD_ERROR_IF(res != RhiResult::success,
                  "Failed to create pipeline for the adstack SizeExpr sizer shader (err: {})", int(res));
      adstack_sizer_pipeline_ = std::move(pipeline);
    }

    // Encode per-task bytecodes and compute per-task metadata sizes.
    std::vector<std::vector<uint8_t>> per_task_bytecodes(adstack_task_indices.size());
    std::vector<size_t> per_task_bytecode_offsets(adstack_task_indices.size());
    std::vector<size_t> per_task_metadata_bytes(adstack_task_indices.size());
    size_t total_bytecode_bytes = 0;
    for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
      size_t ti = adstack_task_indices[k];
      per_task_bytecodes[k] = encode_adstack_size_expr_device_bytecode_for_spirv(task_attribs[ti].ad_stack,
                                                                                 program_impl_->program, &host_ctx);
      per_task_bytecode_offsets[k] = total_bytecode_bytes;
      total_bytecode_bytes += per_task_bytecodes[k].size();
      per_task_metadata_bytes[k] = (2u + 2u * task_attribs[ti].ad_stack.allocas.size()) * sizeof(uint32_t);
    }

    // Grow the shared bytecode scratch buffer if the concatenated blob outgrew it. Amortised doubling so
    // steady-state launches see no allocation traffic.
    if (!adstack_sizer_bytecode_buffer_ || adstack_sizer_bytecode_buffer_size_ < total_bytecode_bytes) {
      size_t new_size = std::max(total_bytecode_bytes, 2 * adstack_sizer_bytecode_buffer_size_);
      auto [buf, res] = device_->allocate_memory_unique(
          {new_size, /*host_write=*/true, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
      QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack sizer bytecode buffer (size={})", new_size);
      if (adstack_sizer_bytecode_buffer_)
        ctx_buffers_.push_back(std::move(adstack_sizer_bytecode_buffer_));
      adstack_sizer_bytecode_buffer_ = std::move(buf);
      adstack_sizer_bytecode_buffer_size_ = new_size;
    }
    {
      void *mapped = nullptr;
      RhiResult map_res = device_->map_range(adstack_sizer_bytecode_buffer_->get_ptr(0), total_bytecode_bytes, &mapped);
      QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack sizer bytecode buffer for upload");
      for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
        std::memcpy(reinterpret_cast<char *>(mapped) + per_task_bytecode_offsets[k], per_task_bytecodes[k].data(),
                    per_task_bytecodes[k].size());
      }
      device_->unmap(*adstack_sizer_bytecode_buffer_);
    }

    // Per-task metadata output buffers. Defer-freed via `ctx_buffers_` after readback so any in-flight writes
    // from the just-synced sizer dispatch can finish draining through the normal cmdlist cleanup path.
    std::vector<DeviceAllocationUnique> per_task_metadata_allocs(adstack_task_indices.size());
    for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
      auto [buf, res] =
          device_->allocate_memory_unique({per_task_metadata_bytes[k], /*host_write=*/false,
                                           /*host_read=*/true, /*export_sharing=*/false, AllocUsage::Storage});
      QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack sizer output buffer (size={})",
                     per_task_metadata_bytes[k]);
      per_task_metadata_allocs[k] = std::move(buf);
    }

    // Force visibility of prior device-side writes (accessor-kernel snode writes, user-side ndarray h2d blits,
    // adstack heap grow-path `buffer_fill`s) to the sizer's `PhysicalStorageBuffer` loads. An intra-cmdlist
    // `memory_barrier()` is not sufficient on MoltenVK: the Metal command encoder backs PSB loads through the
    // device-address path, which bypasses the descriptor-bound cache a prior accessor kernel's `submit_synced`
    // flushed via `vkQueueWaitIdle`. `flush()` drains any pending `current_cmdlist_` the outer launcher may have
    // left behind (the one the main-kernel dispatch below will reuse), and `vkDeviceWaitIdle` pairs with the
    // queue-level fence semantics the MoltenVK driver honours for cross-memory-path coherency. Symptom without
    // this: `FieldLoad(n_iter) -> 0` instead of the live field value, then an adstack overflow at the next
    // `qd.sync()`.
    flush();
    device_->wait_idle();
    auto [sizer_cmdlist, cmdlist_res] = device_->get_compute_stream()->new_command_list_unique();
    QD_ASSERT_INFO(cmdlist_res == RhiResult::success, "Failed to create adstack sizer cmdlist");

    for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
      auto bindings = device_->create_resource_set_unique();
      // All three bindings are declared as `StorageBuffer` in the sizer shader (`buffer_argument` lowers to
      // a storage SSBO in the SPIR-V IR, not a uniform). Vulkan distinguishes uniform and storage buffers
      // via distinct `VkDescriptorType` values - binding these slots via `buffer()` (uniform) produces a
      // descriptor set layout that doesn't match the pipeline's, and `bind_shader_resources` returns
      // `invalid_usage` with "Layout mismatch". Use `rw_buffer` across the board so the descriptor types
      // match; the shader is disciplined about not writing slots 0 and 2, so granting write-capable
      // descriptors there is harmless.
      bindings->rw_buffer(0, adstack_sizer_bytecode_buffer_->get_ptr(per_task_bytecode_offsets[k]),
                          per_task_bytecodes[k].size());
      bindings->rw_buffer(1, *per_task_metadata_allocs[k]);
      // Buffer(2) holds the outer kernel's arg buffer: the sizer reads ndarray data pointers out of it to
      // resolve `ExternalTensorRead` nodes. A kernel can legitimately have adstack allocas *without* any
      // ndarray-backed inputs (e.g. adstacks sized from a field value only, not from an ndarray shape),
      // in which case `args_buffer` is null and no ExternalTensorRead nodes ever get interpreted - so any
      // valid allocation is safe to bind here. Fall back to the bytecode buffer rather than plumbing a
      // conditional null-binding path that every RHI backend would need to support.
      bindings->rw_buffer(2, args_buffer ? *args_buffer : *adstack_sizer_bytecode_buffer_);

      sizer_cmdlist->bind_pipeline(adstack_sizer_pipeline_.get());
      RhiResult bind_res = sizer_cmdlist->bind_shader_resources(bindings.get());
      QD_ERROR_IF(bind_res != RhiResult::success, "Sizer resource binding error: RhiResult({})", int(bind_res));
      RhiResult dispatch_res = sizer_cmdlist->dispatch(1, 1, 1);
      QD_ERROR_IF(dispatch_res != RhiResult::success, "Sizer dispatch error: RhiResult({})", int(dispatch_res));
      sizer_cmdlist->buffer_barrier(*per_task_metadata_allocs[k]);
    }
    device_->get_compute_stream()->submit_synced(sizer_cmdlist.get());

    for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
      size_t ti = adstack_task_indices[k];
      auto &rt = per_task_ad_stack[ti];
      const size_t n_u32 = per_task_metadata_bytes[k] / sizeof(uint32_t);
      rt.metadata.resize(n_u32);
      void *mapped = nullptr;
      RhiResult map_res = device_->map(*per_task_metadata_allocs[k], &mapped);
      QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack sizer output buffer for readback");
      std::memcpy(rt.metadata.data(), mapped, per_task_metadata_bytes[k]);
      device_->unmap(*per_task_metadata_allocs[k]);
      rt.stride_float = rt.metadata[0];
      rt.stride_int = rt.metadata[1];
      // `QD_DEBUG_ADSTACK=1` opt-in diagnostic. Dumps the encoded bytecode's per-stack header (root_node_idx,
      // max_size_compile_time, heap_kind, entry_size_bytes) alongside the runtime-evaluated (offset,
      // max_size) that the sizer shader wrote back. A `root_node_idx < 0` stack means the host encoder
      // found no symbolic SizeExpr for the alloca (empty `size_expr.nodes`), so the sizer falls back to
      // `max_size_compile_time` - that's the most common Genesis overflow cause and it's otherwise
      // invisible. Printed to stderr, one line per stack, unconditional when the env var is set.
      if (std::getenv("QD_DEBUG_ADSTACK")) {
        const auto &bc = per_task_bytecodes[k];
        const auto *hdr = reinterpret_cast<const AdStackSizeExprDeviceHeader *>(bc.data());
        const auto *stack_headers =
            reinterpret_cast<const AdStackSizeExprDeviceStackHeader *>(bc.data() + sizeof(AdStackSizeExprDeviceHeader));
        std::fprintf(stderr,
                     "[adstack_sizer] kernel='%s' task=%zu allocas=%zu bytecode_bytes=%zu "
                     "n_stacks=%u total_nodes=%u total_indices=%u stride_f=%u stride_i=%u\n",
                     ti_kernel->ti_kernel_attribs().name.c_str(), ti, task_attribs[ti].ad_stack.allocas.size(),
                     bc.size(), hdr->n_stacks, hdr->total_nodes, hdr->total_indices, rt.stride_float, rt.stride_int);
        for (uint32_t si = 0; si < hdr->n_stacks; ++si) {
          const auto &sh = stack_headers[si];
          uint32_t off = rt.metadata[2 + 2 * si];
          uint32_t mx = rt.metadata[2 + 2 * si + 1];
          std::fprintf(stderr,
                       "[adstack_sizer]   stack[%u]: heap=%s entry_bytes=%u root_idx=%d max_size_ct=%u -> offset=%u "
                       "max_size=%u%s\n",
                       si, sh.heap_kind == 0 ? "F" : "I", sh.entry_size_bytes, sh.root_node_idx,
                       sh.max_size_compile_time, off, mx,
                       sh.root_node_idx < 0 ? " [fallback to compile-time bound - no symbolic tree]" : "");
        }
      }
      // Sanity cap: a per-thread adstack stride larger than this indicates the sizer shader returned garbage
      // (e.g. an `ExternalTensorRead` dereferenced an uninitialised arg-buffer slot). Without this guard the
      // downstream heap allocation below multiplies by `dispatched_threads` and asks the RHI for hundreds of
      // GB, which tears the machine down with OOM before any error surface has a chance to run. 16 Mi u32 words
      // per thread is already far beyond any realistic reverse-mode workload; pin it and hard-error so the bug
      // is attributed to the sizer output, not to the heap allocator at the call site that used the result.
      constexpr uint32_t kMaxSaneStridePerThread = 1u << 24;
      QD_ERROR_IF(rt.stride_float > kMaxSaneStridePerThread || rt.stride_int > kMaxSaneStridePerThread,
                  "Adstack sizer shader returned an implausibly large per-thread stride (stride_float={}, "
                  "stride_int={}, cap={}). This is almost always a bug in `encode_adstack_size_expr_device_"
                  "bytecode_for_spirv` (wrong `kNodeOffArgBufferOffset` or missing `ExternalTensorRead` "
                  "pre-substitution) or in the sizer shader's PSB read path, not a legitimate workload.",
                  rt.stride_float, rt.stride_int, kMaxSaneStridePerThread);
      ctx_buffers_.push_back(std::move(per_task_metadata_allocs[k]));
    }
  }

  ensure_current_cmdlist();

  for (int i = 0; i < task_attribs.size(); ++i) {
    const auto &attribs = task_attribs[i];
    auto vp = ti_kernel->get_pipeline(i);

    // Cap `advisory_total_num_threads` to the ACTUAL iteration count when the codegen was able to extract the range
    // end as a product of ndarray-shape lookups (see `RangeForAttributes::end_shape_product`). Without this cap, a
    // grad kernel whose range is runtime-determined (`const_end = false`) inherits `kMaxNumThreadsGridStrideLoop =
    // 131072` from the codegen fallback, and the adstack-heap sizing below multiplies that by the per-thread stride
    // to request (e.g.) 48 GB for a 1-iteration B=1 workload - exceeding Metal's `maxBufferLength` and producing a
    // hard RHI error. The in-shader grid-stride loop handles any dispatched thread count >= 1 correctly; a tight cap
    // just means each dispatched thread processes fewer strides of idle work.
    int effective_advisory_threads = attribs.advisory_total_num_threads;
    if (attribs.range_for_attribs && !attribs.range_for_attribs->end_shape_product.empty()) {
      const auto &range = *attribs.range_for_attribs;
      // `const_begin` is asserted true at codegen whenever `end_stmt` is populated (see the
      // `QD_ASSERT(stmt->const_begin)` in the `if (stmt->end_stmt)` branch of spirv_codegen.cpp,
      // near line 1833 at time of writing), so `range.begin` is the literal begin value, not a
      // gtmp offset.
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
        // SPIR-V codegen writes a non-zero sentinel into this single-u32 buffer whenever an AdStackPushStmt hits
        // the overflow branch. Allocate it lazily on first use and reuse across launches; synchronize() reads it,
        // raises on non-zero, and zeros it for the next window.
        if (!adstack_overflow_buffer_) {
          auto [buf, res] = device_->allocate_memory_unique({sizeof(uint32_t), /*host_write=*/true, /*host_read=*/true,
                                                             /*export_sharing=*/false, AllocUsage::Storage});
          QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack overflow buffer");
          adstack_overflow_buffer_ = std::move(buf);
          current_cmdlist_->buffer_fill(adstack_overflow_buffer_->get_ptr(0), kBufferSizeEntireSize, /*data=*/0);
          current_cmdlist_->buffer_barrier(*adstack_overflow_buffer_);
        }
        bindings->rw_buffer(bind.binding, *adstack_overflow_buffer_);
      } else if (bind.buffer.type == BufferType::AdStackHeapFloat) {
        // SPIR-V adstack primal/adjoint storage for f32 adstacks. Sized for the actual dispatched thread count
        // (`group_x * block_dim`, which rounds `advisory_total_num_threads` up to a workgroup multiple) rather
        // than the advisory so threads past the advisory - which still own an `invoc_id * stride` slice - stay
        // in-bounds even if they ever reach a push/pop. Grown on demand and reused across launches; contents do
        // not need to persist across kernels. On empty fields (`dispatched_threads == 0`) no push/pop can
        // actually execute, so bind a null allocation instead of asking the RHI for a zero-sized buffer (which
        // trips `RHI_ASSERT(params.size > 0)` on Vulkan and fails similarly on Metal). The stride used here is
        // the per-launch value produced by `evaluate_adstack_size_expr` over every alloca (stored in
        // `ad_stack_stride_float`), not the compile-time `attribs.ad_stack.per_thread_stride_float_compile_time`.
        size_t dispatched_threads = size_t(group_x) * size_t(attribs.advisory_num_threads_per_group);
        // The shader uses u64 index arithmetic for `invoc_id * stride + offset + count` when the device has
        // Int64; without Int64 the shader falls back to u32 OpIMul, which silently wraps past 2^32 and aliases
        // threads into one another's heap slice. Assert at launch time rather than emit silent corruption.
        QD_ASSERT_INFO(device_->get_caps().get(DeviceCapability::spirv_has_int64) ||
                           size_t(ad_stack_stride_float) * dispatched_threads <= std::numeric_limits<uint32_t>::max(),
                       "adstack f32 heap offset would overflow u32 on a device without Int64: "
                       "stride={} dispatched_threads={}",
                       ad_stack_stride_float, dispatched_threads);
        size_t required = size_t(ad_stack_stride_float) * dispatched_threads * sizeof(float);
        if (required == 0) {
          bindings->rw_buffer(bind.binding, kDeviceNullAllocation);
        } else {
          if (!adstack_heap_buffer_float_ || adstack_heap_buffer_float_size_ < required) {
            // Amortized doubling: mirrors `LlvmRuntimeExecutor::ensure_adstack_heap`. Without it, a sequence of
            // launches with monotonically increasing dispatch sizes (e.g. BFS / frontier expansion) between
            // `synchronize()` calls would reallocate on every launch and leave every displaced buffer sitting in
            // `ctx_buffers_` until the next sync, accumulating O(K^2 * N) bytes of live-but-unused GPU memory.
            // Doubling bounds the reallocations at O(log K) and the live memory at O(K * N).
            size_t new_size = std::max(required, 2 * adstack_heap_buffer_float_size_);
            auto [buf, res] = device_->allocate_memory_unique(
                {new_size, /*host_write=*/false, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
            // Fallback when the amortized-doubling size overshoots a device limit (e.g. Metal's
            // `maxBufferLength` capping `2 * old_size` even when `required` alone would fit): retry at exactly
            // `required` bytes before aborting the process. Trade-off is losing amortization on the retry path;
            // still correct because the next grow will reset amortization against the new, smaller base.
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
        // from `ad_stack_metadata` above (which evaluated each alloca's `SizeExpr`). Empty `allocas` means the
        // codegen never emitted a `BufferType::AdStackMetadata` bind in the first place, so this branch is
        // reached only when we have something to upload. Sized and grown with the same amortized-doubling
        // retire-old-into-`ctx_buffers_` policy as the heap buffers.
        QD_ASSERT_INFO(!ad_stack_metadata.empty(),
                       "AdStackMetadata bind requested for a task that recorded no adstack allocas");
        const size_t required = ad_stack_metadata.size() * sizeof(uint32_t);
        if (!adstack_metadata_buffer_ || adstack_metadata_buffer_size_ < required) {
          size_t new_size = std::max(required, 2 * adstack_metadata_buffer_size_);
          auto [buf, res] = device_->allocate_memory_unique(
              {new_size, /*host_write=*/true, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
          if (res != RhiResult::success && new_size > required) {
            new_size = required;
            std::tie(buf, res) = device_->allocate_memory_unique(
                {new_size, /*host_write=*/true, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
          }
          QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack metadata buffer (size={})", new_size);
          if (adstack_metadata_buffer_) {
            ctx_buffers_.push_back(std::move(adstack_metadata_buffer_));
          }
          adstack_metadata_buffer_ = std::move(buf);
          adstack_metadata_buffer_size_ = new_size;
        }
        // Map, memcpy, unmap. Upload happens on the host timeline; the shader dispatch below happens after
        // the cmdlist is submitted, so a host-side write is visible to the device without an explicit barrier
        // on Vulkan (host-visible memory has implicit HOST -> DEVICE availability at submit time) and on
        // Metal (the shared buffer's contents are observable after `commit`).
        void *mapped = nullptr;
        RhiResult map_res = device_->map_range(adstack_metadata_buffer_->get_ptr(0), required, &mapped);
        QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack metadata buffer for host upload (size={})",
                       required);
        std::memcpy(mapped, ad_stack_metadata.data(), required);
        device_->unmap(*adstack_metadata_buffer_);
        bindings->rw_buffer(bind.binding, *adstack_metadata_buffer_);
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
  // Async adstack-overflow report: every launch in this sync window that overflowed wrote a non-zero sentinel into
  // the shared flag buffer. Read it now, raise if any kernel overflowed, and zero it so the next sync window starts
  // clean. This mirrors the CUDA async-error pattern: the error surfaces on the next synchronize() rather than per
  // launch. The map() here must stay after the `wait_idle()` above; otherwise a future refactor could reorder and
  // we would race against pending GPU writes.
  if (adstack_overflow_buffer_ && !finalizing_) {
    uint32_t flag_val = 0;
    void *mapped = nullptr;
    QD_ASSERT(device_->map(*adstack_overflow_buffer_, &mapped) == RhiResult::success);
    flag_val = *reinterpret_cast<const uint32_t *>(mapped);
    if (flag_val != 0) {
      *reinterpret_cast<uint32_t *>(mapped) = 0;
    }
    device_->unmap(*adstack_overflow_buffer_);
    QD_ERROR_IF(flag_val != 0,
                "Adstack overflow (offending stack_id={}): a reverse-mode autodiff kernel pushed more elements "
                "than the adstack capacity allows. Raised at the next qd.sync() rather than at the offending "
                "kernel launch. The pre-pass resolved this alloca to a bound tighter than the actual runtime "
                "push count - either the enclosing loop shape is outside the current `SizeExpr` grammar "
                "(rewrite it, or extend the grammar), or the Bellman-Ford analyzer undercounted the "
                "forward-pass accumulation on this stack (file a bug with the kernel IR via `QD_DUMP_IR=1`).",
                flag_val - 1);
  }
  fflush(stdout);
}

StreamSemaphore GfxRuntime::flush() {
  StreamSemaphore sema;
  if (current_cmdlist_) {
    sema = device_->get_compute_stream()->submit(current_cmdlist_.get());
    current_cmdlist_ = nullptr;
    // Do NOT clear ctx_buffers_ here: submit() returns as soon as the cmdlist is queued, not when the GPU has
    // finished executing. The deferred-free buffers in ctx_buffers_ (e.g. the old adstack heap buffer left over
    // after a grow-on-demand resize) may still be referenced by commands in flight. Only `synchronize()` clears
    // the vector, after `wait_idle()` has drained the stream.
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
  // If we have accumulated some work but does not require sync
  // and if the accumulated cmdlist has been pending for some time
  // launch the cmdlist to start processing.
  if (current_cmdlist_) {
    constexpr uint64_t max_pending_time = 2000;  // 2000us = 2ms
    auto duration = high_res_clock::now() - current_cmdlist_pending_since_;
    if (std::chrono::duration_cast<std::chrono::microseconds>(duration).count() > max_pending_time) {
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
