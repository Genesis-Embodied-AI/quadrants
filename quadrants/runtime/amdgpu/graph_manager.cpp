#include "quadrants/runtime/amdgpu/graph_manager.h"
#include "quadrants/runtime/amdgpu/amdgpu_utils.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"

#include <algorithm>
#include <vector>

namespace quadrants::lang {
namespace amdgpu {

CachedGraph::CachedGraph(std::size_t arg_buf_size, std::size_t result_buf_size, LlvmRuntimeExecutor *executor)
    : arg_buffer_size(arg_buf_size), result_buffer_size(result_buf_size) {
  AMDGPUDriver::get_instance().malloc(reinterpret_cast<void **>(&persistent_device_result_buffer),
                                      std::max(result_buffer_size, sizeof(uint64)));

  if (arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().malloc(reinterpret_cast<void **>(&persistent_device_arg_buffer), arg_buffer_size);
  }

  // Device-side `RuntimeContext` for graph kernel-node args. Unlike the CUDA
  // path which passes a host pointer (relying on UVA / HMM), AMDGPU kernels
  // dereference the pointer directly on the GPU, so it must point at device
  // memory. See `runtime/amdgpu/graph_manager.h` for the full rationale.
  AMDGPUDriver::get_instance().malloc(&device_runtime_ctx, sizeof(RuntimeContext));

  persistent_ctx.runtime = executor->get_llvm_runtime();
  persistent_ctx.arg_buffer = persistent_device_arg_buffer;
  persistent_ctx.result_buffer = reinterpret_cast<uint64 *>(persistent_device_result_buffer);
  persistent_ctx.cpu_thread_id = 0;
}

CachedGraph::~CachedGraph() {
  if (graph_exec) {
    AMDGPUDriver::get_instance().graph_exec_destroy(graph_exec);
  }
  if (persistent_device_arg_buffer) {
    AMDGPUDriver::get_instance().mem_free(persistent_device_arg_buffer);
  }
  if (persistent_device_result_buffer) {
    AMDGPUDriver::get_instance().mem_free(persistent_device_result_buffer);
  }
  if (device_runtime_ctx) {
    AMDGPUDriver::get_instance().mem_free(device_runtime_ctx);
  }
}

CachedGraph::CachedGraph(CachedGraph &&other) noexcept
    : graph_exec(other.graph_exec),
      persistent_device_arg_buffer(other.persistent_device_arg_buffer),
      persistent_device_result_buffer(other.persistent_device_result_buffer),
      persistent_ctx(other.persistent_ctx),
      device_runtime_ctx(other.device_runtime_ctx),
      arg_buffer_size(other.arg_buffer_size),
      result_buffer_size(other.result_buffer_size),
      num_nodes(other.num_nodes) {
  other.graph_exec = nullptr;
  other.persistent_device_arg_buffer = nullptr;
  other.persistent_device_result_buffer = nullptr;
  other.device_runtime_ctx = nullptr;
}

CachedGraph &CachedGraph::operator=(CachedGraph &&other) noexcept {
  // Move-and-swap: after the swaps, `raii_guard` holds our old resources and
  // its destructor frees them, so every owned pointer is released uniformly.
  CachedGraph raii_guard(std::move(other));
  std::swap(graph_exec, raii_guard.graph_exec);
  std::swap(persistent_device_arg_buffer, raii_guard.persistent_device_arg_buffer);
  std::swap(persistent_device_result_buffer, raii_guard.persistent_device_result_buffer);
  std::swap(persistent_ctx, raii_guard.persistent_ctx);
  std::swap(device_runtime_ctx, raii_guard.device_runtime_ctx);
  std::swap(arg_buffer_size, raii_guard.arg_buffer_size);
  std::swap(result_buffer_size, raii_guard.result_buffer_size);
  std::swap(num_nodes, raii_guard.num_nodes);
  return *this;
}

// Resolves ndarray parameter handles in the launch context to raw device
// pointers, writing them into the arg buffer via `set_ndarray_ptrs`.
//
// Unlike the normal launch path, this does not handle host-resident arrays
// (no temporary device allocation or host-to-device transfer). Errors if any
// external array is on the host, since graph mode bakes device pointers into
// the cached graph.
void GraphManager::resolve_ctx_ndarray_ptrs(LaunchContextBuilder &ctx,
                                            const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                                            LlvmRuntimeExecutor *executor) {
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

      QD_ERROR_IF(grad_ptr != nullptr,
                  "graph does not support autograd; "
                  "ndarray arg {} has a non-null gradient pointer",
                  arg_id);

      void *resolved_data = nullptr;
      if (ctx.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone) {
        QD_ERROR_IF(!on_amdgpu_device(data_ptr),
                    "graph requires all ndarrays to be device-resident; "
                    "ndarray arg {} is host-resident",
                    arg_id);
        resolved_data = data_ptr;
      } else if (arr_sz > 0) {
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        resolved_data = executor->get_device_alloc_info_ptr(*ptr);
      }

      if (resolved_data) {
        ctx.set_ndarray_ptrs(arg_id, (uint64)resolved_data, (uint64) nullptr);
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = resolved_data;
        }
      }
    }
  }
}

void *GraphManager::add_kernel_node(void *graph,
                                    void *prev_node,
                                    void *func,
                                    unsigned int grid_dim,
                                    unsigned int block_dim,
                                    unsigned int shared_mem,
                                    void **kernel_params) {
  // Opt in to the requested dynamic shared memory size. `hipFuncSetAttribute`
  // is mostly a no-op for the AMD backend per its header comments, but we
  // call it for parity with the CUDA path and to forward the request to any
  // backend that honours it.
  if (shared_mem > 0) {
    AMDGPUDriver::get_instance().kernel_set_attribute(func, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                      static_cast<int>(shared_mem));
  }

  HipKernelNodeParams params{};
  params.blockDimX = block_dim;
  params.blockDimY = 1;
  params.blockDimZ = 1;
  params.extra = nullptr;
  params.func = func;
  params.gridDimX = grid_dim;
  params.gridDimY = 1;
  params.gridDimZ = 1;
  params.kernelParams = kernel_params;
  params.sharedMemBytes = shared_mem;

  void *node = nullptr;
  AMDGPUDriver::get_instance().graph_add_kernel_node(&node, graph, prev_node ? &prev_node : nullptr, prev_node ? 1 : 0,
                                                     &params);
  return node;
}

bool GraphManager::launch_cached_graph(CachedGraph &cached, LaunchContextBuilder &ctx) {
  if (ctx.arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().memcpy_host_to_device(cached.persistent_device_arg_buffer,
                                                       ctx.get_context().arg_buffer, cached.arg_buffer_size);
  }
  auto *stream = AMDGPUContext::get_instance().get_stream();
  AMDGPUDriver::get_instance().graph_launch(cached.graph_exec, stream);
  used_on_last_call_ = true;
  num_nodes_on_last_call_ = cached.num_nodes;
  return true;
}

bool GraphManager::try_launch(int launch_id,
                              LaunchContextBuilder &ctx,
                              JITModule *amdgpu_module,
                              const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                              const std::vector<OffloadedTask> &offloaded_tasks,
                              LlvmRuntimeExecutor *executor) {
  if (offloaded_tasks.empty()) {
    return false;
  }

  QD_ERROR_IF(ctx.result_buffer_size > 0,
              "graph=True is not supported for kernels with struct return "
              "values; remove graph=True or avoid returning values");

  // Adstack-bearing kernels cannot go through the graph path. See the matching comment in
  // `runtime/cuda/graph_manager.cpp::try_launch` for the full rationale: the per-task adstack
  // setup runs host-side between the serial range_for-bounds kernel and the range_for kernel
  // itself, and there is no host hook between graph nodes.
  for (const auto &task : offloaded_tasks) {
    QD_ERROR_IF(!task.ad_stack.allocas.empty(),
                "graph=True is not supported for kernels that use the reverse-mode autodiff stack "
                "(task '{}' has {} adstack allocas). Launch without graph=True.",
                task.name, task.ad_stack.allocas.size());
  }

  resolve_ctx_ndarray_ptrs(ctx, parameters, executor);

  auto it = cache_.find(launch_id);
  if (it != cache_.end()) {
    return launch_cached_graph(it->second, ctx);
  }

  AMDGPUContext::get_instance().make_current();
  CachedGraph cached(ctx.arg_buffer_size, ctx.result_buffer_size, executor);

  if (cached.arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().memcpy_host_to_device(cached.persistent_device_arg_buffer,
                                                       ctx.get_context().arg_buffer, cached.arg_buffer_size);
  }

  // Stage the RuntimeContext on device. Its arg_buffer / result_buffer
  // pointers reference the persistent device buffers above; none of its
  // fields change between graph launches so one copy is sufficient.
  AMDGPUDriver::get_instance().memcpy_host_to_device(cached.device_runtime_ctx, &cached.persistent_ctx,
                                                     sizeof(RuntimeContext));

  void *graph = nullptr;
  AMDGPUDriver::get_instance().graph_create(&graph, 0);

  // Each kernel node receives the device-side RuntimeContext pointer. Note we
  // bake the value of `device_runtime_ctx` (a single device address) into the
  // kernel-node args; the kernels on the GPU then dereference this pointer to
  // read the arg buffer / result buffer pointers. Stream-parallel groups
  // (`stream_parallel_group_id != 0`) are silently serialized inside the
  // graph, matching the CUDA implementation.
  void *prev_node = nullptr;
  for (const auto &task : offloaded_tasks) {
    void *ctx_ptr = cached.device_runtime_ctx;
    prev_node =
        add_kernel_node(graph, prev_node, amdgpu_module->lookup_function(task.name), (unsigned int)task.grid_dim,
                        (unsigned int)task.block_dim, (unsigned int)task.dynamic_shared_array_bytes, &ctx_ptr);
  }

  AMDGPUDriver::get_instance().graph_instantiate(&cached.graph_exec, graph, nullptr, nullptr, 0);

  auto *stream = AMDGPUContext::get_instance().get_stream();
  AMDGPUDriver::get_instance().graph_launch(cached.graph_exec, stream);

  AMDGPUDriver::get_instance().graph_destroy(graph);

  cached.num_nodes = offloaded_tasks.size();
  QD_TRACE("HIP graph created with {} kernel nodes for launch_id={}", cached.num_nodes, launch_id);

  num_nodes_on_last_call_ = cached.num_nodes;
  ++total_builds_;
  cache_.emplace(launch_id, std::move(cached));
  used_on_last_call_ = true;
  return true;
}

}  // namespace amdgpu
}  // namespace quadrants::lang
