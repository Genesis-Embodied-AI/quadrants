#include "quadrants/runtime/amdgpu/kernel_launcher.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/runtime/llvm/llvm_runtime_executor.h"

namespace quadrants::lang {
namespace amdgpu {

namespace {

// Resolve the adstack thread count this task needs sizing for.
//
// For const-bound range_for and non-range_for tasks, codegen has already made `static_num_threads` tight
// (`grid_dim * block_dim` with `grid_dim` clamped to `ceil((end-begin)/block_dim)` for const range_for), so
// we return it directly.
//
// For dynamic-bound range_for tasks, resolve `end - begin` by reading the values codegen stashed into
// `runtime->temporaries` via a host-side DtoH memcpy. Mirrors `runtime/cuda/kernel_launcher.cpp`.
std::size_t resolve_num_threads(const OffloadedTask &task, LlvmRuntimeExecutor *executor) {
  if (!task.ad_stack.dynamic_gpu_range_for) {
    return task.ad_stack.static_num_threads;
  }
  const auto &info = task.ad_stack;
  std::int32_t begin = info.begin_const_value;
  std::int32_t end = info.end_const_value;
  if (info.begin_offset_bytes >= 0 || info.end_offset_bytes >= 0) {
    auto *active_stream = AMDGPUContext::get_instance().get_stream();
    auto *temp_dev_ptr = reinterpret_cast<uint8_t *>(executor->get_runtime_temporaries_device_ptr());
    if (info.begin_offset_bytes >= 0) {
      AMDGPUDriver::get_instance().memcpy_device_to_host_async(&begin, temp_dev_ptr + info.begin_offset_bytes,
                                                               sizeof(std::int32_t), active_stream);
    }
    if (info.end_offset_bytes >= 0) {
      AMDGPUDriver::get_instance().memcpy_device_to_host_async(&end, temp_dev_ptr + info.end_offset_bytes,
                                                               sizeof(std::int32_t), active_stream);
    }
    AMDGPUDriver::get_instance().stream_synchronize(active_stream);
  }
  // Clamp the logical iteration count to the launched thread count: adstack slices are indexed by
  // `linear_thread_idx()`, so only `static_num_threads = grid_dim * block_dim` slices can be touched
  // concurrently. See the matching comment in `runtime/cuda/kernel_launcher.cpp`.
  std::size_t iter = end > begin ? static_cast<std::size_t>(end - begin) : 0;
  return std::min(iter, task.ad_stack.static_num_threads);
}

}  // namespace

void KernelLauncher::launch_offloaded_tasks(LaunchContextBuilder &ctx,
                                            JITModule *amdgpu_module,
                                            const std::vector<OffloadedTask> &offloaded_tasks,
                                            void *context_pointer,
                                            int arg_size) {
  auto *executor = get_runtime_executor();
  for (const auto &task : offloaded_tasks) {
    // Pass the device-side `RuntimeContext` pointer through to the adstack sizer kernel. Without this the
    // sizer launches with a host pointer and the next DtoH sync trips
    // `hipErrorIllegalAddress ... memcpy_device_to_host` because HIP has no UVA fallback for the host
    // `RuntimeContext` struct.
    executor->publish_adstack_metadata(task.ad_stack, resolve_num_threads(task, executor), &ctx, context_pointer);
    QD_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim, task.block_dim);
    amdgpu_module->launch(task.name, task.grid_dim, task.block_dim, task.dynamic_shared_array_bytes,
                          {(void *)&context_pointer}, {arg_size});
  }
}

void KernelLauncher::launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx,
                                                          JITModule *amdgpu_module,
                                                          const std::vector<OffloadedTask> &offloaded_tasks,
                                                          void *context_pointer,
                                                          int arg_size) {
  int32_t counter_val;
  do {
    launch_offloaded_tasks(ctx, amdgpu_module, offloaded_tasks, context_pointer, arg_size);
    counter_val = 0;
    auto *stream = AMDGPUContext::get_instance().get_stream();
    AMDGPUDriver::get_instance().stream_synchronize(stream);
    AMDGPUDriver::get_instance().memcpy_device_to_host(&counter_val, ctx.graph_do_while_flag_dev_ptr, sizeof(int32_t));
  } while (counter_val != 0);
}

bool KernelLauncher::on_amdgpu_device(void *ptr) {
  unsigned int attr_val[8];
  // mem_get_attribute doesn't work well on ROCm
  uint32_t ret_code = AMDGPUDriver::get_instance().mem_get_attributes.call(attr_val, ptr);

  return ret_code == HIP_SUCCESS && attr_val[0] == HIP_MEMORYTYPE_DEVICE;
}

void KernelLauncher::launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) {
  QD_ASSERT(handle.get_launch_id() < contexts_.size());
  auto launcher_ctx = contexts_[handle.get_launch_id()];
  auto *executor = get_runtime_executor();
  auto *amdgpu_module = launcher_ctx.jit_module;
  const auto &parameters = *launcher_ctx.parameters;
  const auto &offloaded_tasks = launcher_ctx.offloaded_tasks;

  AMDGPUContext::get_instance().make_current();
  ctx.get_context().runtime = executor->get_llvm_runtime();

  // Change from std::vector<int> to ArgArrayPtrKey
  std::unordered_map<ArgArrayPtrKey, std::pair<void *, DeviceAllocation>, ArgArrayPtrKeyHasher> transfers;
  std::unordered_map<ArgArrayPtrKey, void *, ArgArrayPtrKeyHasher> device_ptrs;

  auto *active_stream = AMDGPUContext::get_instance().get_stream();

  char *device_result_buffer{nullptr};
  // Must always allocate device_result_buffer (even when result_buffer_size is 0) to avoid memory access faults
  // from allocate_memory_on_device below.
  AMDGPUDriver::get_instance().malloc_async((void **)&device_result_buffer,
                                            std::max(ctx.result_buffer_size, sizeof(uint64)), active_stream);

  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &arg_id = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[arg_id];
      if (arr_sz == 0)
        continue;

      ArgArrayPtrKey data_ptr_idx{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      ArgArrayPtrKey grad_ptr_idx{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
      auto data_ptr = ctx.array_ptrs[data_ptr_idx];
      auto grad_ptr = ctx.array_ptrs[grad_ptr_idx];

      if (ctx.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone) {
        // External array. Note: assuming both data & grad are on the same device.
        if (on_amdgpu_device(data_ptr)) {
          device_ptrs[data_ptr_idx] = data_ptr;
          device_ptrs[grad_ptr_idx] = grad_ptr;
        } else {
          DeviceAllocation devalloc = executor->allocate_memory_on_device(arr_sz, (uint64 *)device_result_buffer);
          device_ptrs[data_ptr_idx] = executor->get_device_alloc_info_ptr(devalloc);
          transfers[data_ptr_idx] = {data_ptr, devalloc};

          AMDGPUDriver::get_instance().memcpy_host_to_device_async((void *)device_ptrs[data_ptr_idx], data_ptr, arr_sz,
                                                                   active_stream);
          if (grad_ptr != nullptr) {
            DeviceAllocation grad_devalloc =
                executor->allocate_memory_on_device(arr_sz, (uint64 *)device_result_buffer);
            device_ptrs[grad_ptr_idx] = executor->get_device_alloc_info_ptr(grad_devalloc);
            transfers[grad_ptr_idx] = {grad_ptr, grad_devalloc};

            AMDGPUDriver::get_instance().memcpy_host_to_device_async((void *)device_ptrs[grad_ptr_idx], grad_ptr,
                                                                     arr_sz, active_stream);
          } else {
            device_ptrs[grad_ptr_idx] = nullptr;
          }
        }
        ctx.set_ndarray_ptrs(arg_id, (uint64)device_ptrs[data_ptr_idx], (uint64)device_ptrs[grad_ptr_idx]);
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = device_ptrs[data_ptr_idx];
        }
      } else if (arr_sz > 0) {  // why use arr_sz constrain?
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
    AMDGPUDriver::get_instance().stream_synchronize(active_stream);
  }
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    ctx.get_context().result_buffer = (uint64 *)device_result_buffer;
  }
  char *device_arg_buffer = nullptr;
  if (ctx.arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().malloc_async((void **)&device_arg_buffer, ctx.arg_buffer_size, active_stream);
    AMDGPUDriver::get_instance().memcpy_host_to_device_async(device_arg_buffer, ctx.get_context().arg_buffer,
                                                             ctx.arg_buffer_size, active_stream);
    ctx.get_context().arg_buffer = device_arg_buffer;
  }
  void *context_pointer;
  int arg_size = sizeof(RuntimeContext *);
  AMDGPUDriver::get_instance().malloc_async((void **)&context_pointer, sizeof(RuntimeContext), active_stream);
  AMDGPUDriver::get_instance().memcpy_host_to_device_async(context_pointer, &ctx.get_context(), sizeof(RuntimeContext),
                                                           active_stream);

  if (ctx.graph_do_while_arg_id >= 0) {
    QD_ASSERT(ctx.graph_do_while_flag_dev_ptr);
    launch_offloaded_tasks_with_do_while(ctx, amdgpu_module, offloaded_tasks, context_pointer, arg_size);
  } else {
    launch_offloaded_tasks(ctx, amdgpu_module, offloaded_tasks, context_pointer, arg_size);
  }
  QD_TRACE("Launching kernel");
  if (ctx.arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().mem_free_async(device_arg_buffer, active_stream);
  }
  if (ctx.result_buffer_size > 0) {
    AMDGPUDriver::get_instance().memcpy_device_to_host_async(host_result_buffer, device_result_buffer,
                                                             ctx.result_buffer_size, active_stream);
  }
  AMDGPUDriver::get_instance().mem_free_async(device_result_buffer, active_stream);
  if (transfers.size()) {
    AMDGPUDriver::get_instance().stream_synchronize(active_stream);
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      auto &idx = itr->first;
      AMDGPUDriver::get_instance().memcpy_device_to_host_async(itr->second.first, (void *)device_ptrs[idx],
                                                               ctx.array_runtime_sizes[idx.arg_id], active_stream);
    }
    AMDGPUDriver::get_instance().stream_synchronize(active_stream);
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      executor->deallocate_memory_on_device(itr->second.second);
    }
  } else if (ctx.result_buffer_size > 0) {
    AMDGPUDriver::get_instance().stream_synchronize(active_stream);
  }
  AMDGPUDriver::get_instance().mem_free_async(context_pointer, active_stream);
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(const LLVM::CompiledKernelData &compiled) {
  QD_ASSERT(compiled.arch() == Arch::amdgpu);

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

}  // namespace amdgpu
}  // namespace quadrants::lang
