#include "quadrants/runtime/cuda/kernel_launcher.h"
#include "quadrants/runtime/cuda/cuda_utils.h"
#include "quadrants/rhi/cuda/cuda_context.h"
#include "quadrants/rhi/cuda/cuda_driver.h"
#include "quadrants/runtime/llvm/llvm_runtime_executor.h"

#include <vector>

namespace quadrants::lang {
namespace cuda {

namespace {

// Resolve the tight thread count for a task's adstack sizing. For dynamic-bound range_for the begin / end
// i32 values live in `runtime->temporaries` on device; the launcher fetches them via a 4-byte DtoH memcpy
// each (dominated by the kernel-launch overhead that follows and only paid for kernels that actually use an
// adstack under a dynamic iteration range). Const-bound range_for and non-range_for tasks use the codegen-
// computed `static_num_threads`.
std::size_t resolve_num_threads(const AdStackSizingInfo &info, LlvmRuntimeExecutor *executor) {
  if (!info.dynamic_gpu_range_for) {
    return info.static_num_threads;
  }
  std::int32_t begin = info.begin_const_value;
  std::int32_t end = info.end_const_value;
  if (info.begin_offset_bytes >= 0 || info.end_offset_bytes >= 0) {
    auto *active_stream = CUDAContext::get_instance().get_stream();
    auto *temp_dev_ptr = reinterpret_cast<uint8_t *>(executor->get_runtime_temporaries_device_ptr());
    if (info.begin_offset_bytes >= 0) {
      CUDADriver::get_instance().memcpy_device_to_host_async(&begin, temp_dev_ptr + info.begin_offset_bytes,
                                                             sizeof(std::int32_t), active_stream);
    }
    if (info.end_offset_bytes >= 0) {
      CUDADriver::get_instance().memcpy_device_to_host_async(&end, temp_dev_ptr + info.end_offset_bytes,
                                                             sizeof(std::int32_t), active_stream);
    }
    CUDADriver::get_instance().stream_synchronize(active_stream);
  }
  // Clamp the logical iteration count to the launched thread count: adstack slices are indexed by
  // `linear_thread_idx()` (`block_idx * block_dim + thread_idx`), so only `static_num_threads = grid_dim *
  // block_dim` slices can ever be touched concurrently. A logical range much larger than the launch size does
  // not need more heap than `static_num_threads * per_thread_stride`; allocating the logical count would
  // over-commit memory and trip OOM paths for no gain.
  std::size_t iter = end > begin ? static_cast<std::size_t>(end - begin) : 0;
  return std::min(iter, info.static_num_threads);
}

}  // namespace

void KernelLauncher::launch_offloaded_tasks(LaunchContextBuilder &ctx,
                                            JITModule *cuda_module,
                                            const std::vector<OffloadedTask> &offloaded_tasks,
                                            void *device_context_ptr) {
  auto *executor = get_runtime_executor();
  for (const auto &task : offloaded_tasks) {
    std::size_t n = resolve_num_threads(task.ad_stack, executor);
    // Pass the device-side `RuntimeContext` pointer through to the adstack sizer kernel. Without it the sizer
    // launches with a host pointer and the next DtoH sync trips `CUDA_ERROR_ILLEGAL_ADDRESS ... memcpy_device_to_host`
    // on GPUs whose driver + kernel cannot coherently access pageable host memory (the HMM capability gated below in
    // `launch_llvm_kernel`). `nullptr` on HMM-capable setups keeps `publish_adstack_metadata`'s host-pointer fast path.
    executor->publish_adstack_metadata(task.ad_stack, n, &ctx, device_context_ptr);
    QD_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim, task.block_dim);
    cuda_module->launch(task.name, task.grid_dim, task.block_dim, task.dynamic_shared_array_bytes, {&ctx.get_context()},
                        {});
  }
}

void KernelLauncher::launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx,
                                                          JITModule *cuda_module,
                                                          const std::vector<OffloadedTask> &offloaded_tasks,
                                                          void *device_context_ptr) {
  int32_t counter_val;
  do {
    launch_offloaded_tasks(ctx, cuda_module, offloaded_tasks, device_context_ptr);
    counter_val = 0;
    auto *stream = CUDAContext::get_instance().get_stream();
    CUDADriver::get_instance().memcpy_device_to_host_async(&counter_val, ctx.graph_do_while_flag_dev_ptr,
                                                           sizeof(int32_t), stream);
    CUDADriver::get_instance().stream_synchronize(stream);
  } while (counter_val != 0);
}

void KernelLauncher::launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) {
  QD_ASSERT(handle.get_launch_id() < contexts_.size());

  if (ctx.use_graph) {
    auto &lctx = contexts_[handle.get_launch_id()];
    if (graph_manager_.try_launch(handle.get_launch_id(), ctx, lctx.jit_module, *lctx.parameters, lctx.offloaded_tasks,
                                  get_runtime_executor())) {
      return;
    }
  }
  graph_manager_.mark_not_used();

  auto launcher_ctx = contexts_[handle.get_launch_id()];
  auto *executor = get_runtime_executor();
  auto *cuda_module = launcher_ctx.jit_module;
  const auto &parameters = *launcher_ctx.parameters;
  const auto &offloaded_tasks = launcher_ctx.offloaded_tasks;

  CUDAContext::get_instance().make_current();

  // |transfers| is only used for external arrays whose data is originally on
  // host. They are first transferred onto device and that device pointer is
  // stored in |device_ptrs| below. |transfers| saves its original pointer so
  // that we can copy the data back once kernel finishes. as well as the
  // temporary device allocations, which can be freed after kernel finishes. Key
  // is [arg_id, ptr_pos], where ptr_pos is TypeFactory::DATA_PTR_POS_IN_NDARRAY
  // for data_ptr and TypeFactory::GRAD_PTR_POS_IN_NDARRAY for grad_ptr. Value
  // is [host_ptr, temporary_device_alloc]. Invariant: temp_devallocs.size() !=
  // 0 <==> transfer happened.
  std::unordered_map<ArgArrayPtrKey, std::pair<void *, DeviceAllocation>, ArgArrayPtrKeyHasher> transfers;

  // |device_ptrs| stores pointers on device for all arrays args, including
  // external arrays and ndarrays, no matter whether the data is originally on
  // device or host.
  // This is the source of truth for us to look for device pointers used in CUDA
  // kernels.
  std::unordered_map<ArgArrayPtrKey, void *, ArgArrayPtrKeyHasher> device_ptrs;

  auto *active_stream = CUDAContext::get_instance().get_stream();

  char *device_result_buffer{nullptr};
  CUDADriver::get_instance().malloc_async((void **)&device_result_buffer,
                                          std::max(ctx.result_buffer_size, sizeof(uint64)), active_stream);
  ctx.get_context().runtime = executor->get_llvm_runtime();

  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &arg_id = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[arg_id];
      // Note: both numpy and PyTorch support arrays/tensors with zeros
      // in shapes, e.g., shape=(0) or shape=(100, 0, 200). This makes
      // `arr_sz` zero.
      if (arr_sz == 0) {
        continue;
      }

      ArgArrayPtrKey data_ptr_idx{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      ArgArrayPtrKey grad_ptr_idx{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
      auto data_ptr = ctx.array_ptrs[data_ptr_idx];
      auto grad_ptr = ctx.array_ptrs[grad_ptr_idx];

      if (ctx.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone) {
        // External array
        // Note: assuming both data & grad are on the same device
        if (on_cuda_device(data_ptr)) {
          // data_ptr is a raw ptr on CUDA device
          device_ptrs[data_ptr_idx] = data_ptr;
          device_ptrs[grad_ptr_idx] = grad_ptr;
        } else {
          DeviceAllocation devalloc = executor->allocate_memory_on_device(arr_sz, (uint64 *)device_result_buffer);
          device_ptrs[data_ptr_idx] = executor->get_device_alloc_info_ptr(devalloc);
          transfers[data_ptr_idx] = {data_ptr, devalloc};

          CUDADriver::get_instance().memcpy_host_to_device_async((void *)device_ptrs[data_ptr_idx], data_ptr, arr_sz,
                                                                 active_stream);
          if (grad_ptr != nullptr) {
            DeviceAllocation grad_devalloc =
                executor->allocate_memory_on_device(arr_sz, (uint64 *)device_result_buffer);
            device_ptrs[grad_ptr_idx] = executor->get_device_alloc_info_ptr(grad_devalloc);
            transfers[grad_ptr_idx] = {grad_ptr, grad_devalloc};

            CUDADriver::get_instance().memcpy_host_to_device_async((void *)device_ptrs[grad_ptr_idx], grad_ptr, arr_sz,
                                                                   active_stream);
          } else {
            device_ptrs[grad_ptr_idx] = nullptr;
          }
        }

        ctx.set_ndarray_ptrs(arg_id, (uint64)device_ptrs[data_ptr_idx], (uint64)device_ptrs[grad_ptr_idx]);
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = device_ptrs[data_ptr_idx];
        }
      } else if (arr_sz > 0) {
        // Ndarray
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        // Unwrapped raw ptr on device
        device_ptrs[data_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);

        if (grad_ptr != nullptr) {
          ptr = static_cast<DeviceAllocation *>(grad_ptr);
          device_ptrs[grad_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);
        } else {
          device_ptrs[grad_ptr_idx] = nullptr;
        }

        ctx.set_ndarray_ptrs(arg_id, (uint64)device_ptrs[data_ptr_idx], (uint64)device_ptrs[grad_ptr_idx]);
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = device_ptrs[data_ptr_idx];
        }
      }
    }
  }
  if (transfers.size() > 0) {
    CUDADriver::get_instance().stream_synchronize(active_stream);
  }
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    ctx.get_context().result_buffer = (uint64 *)device_result_buffer;
  }
  char *device_arg_buffer = nullptr;
  if (ctx.arg_buffer_size > 0) {
    CUDADriver::get_instance().malloc_async((void **)&device_arg_buffer, ctx.arg_buffer_size, active_stream);
    CUDADriver::get_instance().memcpy_host_to_device_async(device_arg_buffer, ctx.get_context().arg_buffer,
                                                           ctx.arg_buffer_size, active_stream);
    ctx.get_context().arg_buffer = device_arg_buffer;
  }

  // Stage a device-side copy of `RuntimeContext` for the adstack sizer kernel on GPUs that cannot dereference plain
  // host pointers (`malloc` / `new`) from device code. CUDA's UVA only covers pinned / managed allocations; the
  // `std::make_unique<RuntimeContext>()` backing is neither. On drivers + kernels that expose HMM / system-allocated
  // memory the sizer can read the host pointer directly and we keep the zero-staging fast path; everywhere else
  // (Turing without HMM, Windows, older Linux without an HMM-capable driver) the host-pointer read faults with
  // `CUDA_ERROR_ILLEGAL_ADDRESS` at the next DtoH sync, so we fall back to a per-launch device copy. Gate is the
  // `PAGEABLE_MEMORY_ACCESS` device attribute rather than a compute-capability threshold: HMM availability is a
  // property of the driver + kernel combo, not the GPU architecture (sm_70 / V100 with a current HMM driver is fine;
  // sm_90 / H100 on a Windows host or with a pre-535 Linux driver still needs the fallback). We additionally skip
  // the staging for forward-only launches (no task has an adstack alloca, so `publish_adstack_metadata` early-returns
  // and the staged buffer would be pure waste).
  bool needs_sizer_device_ctx = false;
  for (const auto &task : offloaded_tasks) {
    if (!task.ad_stack.allocas.empty()) {
      needs_sizer_device_ctx = true;
      break;
    }
  }
  needs_sizer_device_ctx = needs_sizer_device_ctx && !CUDAContext::get_instance().supports_pageable_memory_access();
  void *device_context_ptr = nullptr;
  if (needs_sizer_device_ctx) {
    CUDADriver::get_instance().malloc_async(&device_context_ptr, sizeof(RuntimeContext), active_stream);
    CUDADriver::get_instance().memcpy_host_to_device_async(device_context_ptr, &ctx.get_context(),
                                                           sizeof(RuntimeContext), active_stream);
  }

  if (ctx.graph_do_while_arg_id >= 0) {
    QD_ASSERT(ctx.graph_do_while_flag_dev_ptr);
    launch_offloaded_tasks_with_do_while(ctx, cuda_module, offloaded_tasks, device_context_ptr);
  } else {
    launch_offloaded_tasks(ctx, cuda_module, offloaded_tasks, device_context_ptr);
  }
  if (needs_sizer_device_ctx) {
    CUDADriver::get_instance().mem_free_async(device_context_ptr, active_stream);
  }
  if (ctx.arg_buffer_size > 0) {
    CUDADriver::get_instance().mem_free_async(device_arg_buffer, active_stream);
  }
  if (ctx.result_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_device_to_host_async(host_result_buffer, device_result_buffer,
                                                           ctx.result_buffer_size, active_stream);
  }
  CUDADriver::get_instance().mem_free_async(device_result_buffer, active_stream);
  // copy data back to host
  if (transfers.size() > 0) {
    CUDADriver::get_instance().stream_synchronize(active_stream);
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      auto &idx = itr->first;
      CUDADriver::get_instance().memcpy_device_to_host_async(itr->second.first, (void *)device_ptrs[idx],
                                                             ctx.array_runtime_sizes[idx.arg_id], active_stream);
    }
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      executor->deallocate_memory_on_device(itr->second.second);
    }
  }
  CUDADriver::get_instance().stream_synchronize(active_stream);
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(const LLVM::CompiledKernelData &compiled) {
  QD_ASSERT(compiled.arch() == Arch::cuda);

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();
    contexts_.resize(index + 1);

    auto &ctx = contexts_[index];
    auto *executor = get_runtime_executor();

    auto data = compiled.get_internal_data().compiled_data.clone();
    auto *jit_module = executor->create_jit_module(std::move(data.module));

    // Populate ctx
    ctx.jit_module = jit_module;
    ctx.parameters = &compiled.get_internal_data().args;
    ctx.offloaded_tasks = std::move(data.tasks);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

}  // namespace cuda
}  // namespace quadrants::lang
