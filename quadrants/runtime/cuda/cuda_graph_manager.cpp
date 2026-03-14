#include "quadrants/runtime/cuda/cuda_graph_manager.h"
#include "quadrants/runtime/cuda/cuda_utils.h"
#include "quadrants/rhi/cuda/cuda_context.h"

namespace quadrants::lang {
namespace cuda {

CachedCudaGraph::~CachedCudaGraph() {
  if (graph_exec) {
    CUDADriver::get_instance().graph_exec_destroy(graph_exec);
  }
  if (persistent_device_arg_buffer) {
    CUDADriver::get_instance().mem_free(persistent_device_arg_buffer);
  }
  if (persistent_device_result_buffer) {
    CUDADriver::get_instance().mem_free(persistent_device_result_buffer);
  }
}

CachedCudaGraph::CachedCudaGraph(CachedCudaGraph &&other) noexcept
    : graph_exec(other.graph_exec),
      persistent_device_arg_buffer(other.persistent_device_arg_buffer),
      persistent_device_result_buffer(other.persistent_device_result_buffer),
      persistent_ctx(other.persistent_ctx),
      arg_buffer_size(other.arg_buffer_size),
      result_buffer_size(other.result_buffer_size) {
  other.graph_exec = nullptr;
  other.persistent_device_arg_buffer = nullptr;
  other.persistent_device_result_buffer = nullptr;
}

CachedCudaGraph &CachedCudaGraph::operator=(CachedCudaGraph &&other) noexcept {
  if (this != &other) {
    if (graph_exec)
      CUDADriver::get_instance().graph_exec_destroy(graph_exec);
    if (persistent_device_arg_buffer)
      CUDADriver::get_instance().mem_free(persistent_device_arg_buffer);
    if (persistent_device_result_buffer)
      CUDADriver::get_instance().mem_free(persistent_device_result_buffer);

    graph_exec = other.graph_exec;
    persistent_device_arg_buffer = other.persistent_device_arg_buffer;
    persistent_device_result_buffer = other.persistent_device_result_buffer;
    persistent_ctx = other.persistent_ctx;
    arg_buffer_size = other.arg_buffer_size;
    result_buffer_size = other.result_buffer_size;

    other.graph_exec = nullptr;
    other.persistent_device_arg_buffer = nullptr;
    other.persistent_device_result_buffer = nullptr;
  }
  return *this;
}

// Resolves ndarray parameter handles in the launch context to raw device
// pointers, writing them into the arg buffer via set_ndarray_ptrs.
//
// Unlike the normal launch path, this does not handle host-resident arrays
// (no temporary device allocation or host-to-device transfer). Returns false
// if any external array is on the host, signaling the caller to fall back
// to the non-graph launch path.
bool CudaGraphManager::resolve_ctx_ndarray_ptrs(
    LaunchContextBuilder &ctx,
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

      if (ctx.device_allocation_type[arg_id] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        if (!on_cuda_device(data_ptr)) {
          return false;
        }
        ctx.set_ndarray_ptrs(arg_id, (uint64)data_ptr, (uint64)grad_ptr);
      } else if (arr_sz > 0) {
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        void *dev_data = executor->get_device_alloc_info_ptr(*ptr);
        void *dev_grad = nullptr;
        if (grad_ptr) {
          dev_grad = executor->get_device_alloc_info_ptr(
              *static_cast<DeviceAllocation *>(grad_ptr));
        }
        ctx.set_ndarray_ptrs(arg_id, (uint64)dev_data, (uint64)dev_grad);
      }
    }
  }
  return true;
}

void *CudaGraphManager::add_kernel_node(void *graph, void *prev_node,
                                        void *func, unsigned int grid_dim,
                                        unsigned int block_dim,
                                        unsigned int shared_mem,
                                        void **kernel_params) {
  CudaKernelNodeParams params{};
  params.func = func;
  params.gridDimX = grid_dim;
  params.gridDimY = 1;
  params.gridDimZ = 1;
  params.blockDimX = block_dim;
  params.blockDimY = 1;
  params.blockDimZ = 1;
  params.sharedMemBytes = shared_mem;
  params.kernelParams = kernel_params;
  params.extra = nullptr;

  void *node = nullptr;
  CUDADriver::get_instance().graph_add_kernel_node(
      &node, graph, prev_node ? &prev_node : nullptr, prev_node ? 1 : 0,
      &params);
  return node;
}

bool CudaGraphManager::launch_cached_graph(CachedCudaGraph &cached,
                                            LaunchContextBuilder &ctx) {
  if (ctx.arg_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_host_to_device(
        cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer,
        cached.arg_buffer_size);
  }
  auto *stream = CUDAContext::get_instance().get_stream();
  CUDADriver::get_instance().graph_launch(cached.graph_exec, stream);
  used_on_last_call_ = true;
  return true;
}

bool CudaGraphManager::try_launch(
    int launch_id, LaunchContextBuilder &ctx, JITModule *cuda_module,
    const std::vector<std::pair<int, Callable::Parameter>> &parameters,
    const std::vector<OffloadedTask> &offloaded_tasks,
    LlvmRuntimeExecutor *executor) {
  if (offloaded_tasks.empty()) {
    return false;
  }

  QD_ERROR_IF(ctx.result_buffer_size > 0,
              "cuda_graph=True is not supported for kernels with struct return "
              "values; remove cuda_graph=True or avoid returning values");

  // Falls back to the normal path if any external array is host-resident,
  // since the graph path cannot perform host-to-device transfers.
  if (!resolve_ctx_ndarray_ptrs(ctx, parameters, executor)) {
    return false;
  }

  auto it = cache_.find(launch_id);
  if (it != cache_.end()) {
    return launch_cached_graph(it->second, ctx);
  }

  CUDAContext::get_instance().make_current();

  CachedCudaGraph cached;

  // --- Allocate persistent buffers ---
  cached.result_buffer_size = std::max(ctx.result_buffer_size, sizeof(uint64));
  CUDADriver::get_instance().malloc(
      (void **)&cached.persistent_device_result_buffer,
      cached.result_buffer_size);

  cached.arg_buffer_size = ctx.arg_buffer_size;
  if (cached.arg_buffer_size > 0) {
    CUDADriver::get_instance().malloc(
        (void **)&cached.persistent_device_arg_buffer, cached.arg_buffer_size);
    CUDADriver::get_instance().memcpy_host_to_device(
        cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer,
        cached.arg_buffer_size);
  }

  // --- Build persistent RuntimeContext ---
  cached.persistent_ctx.runtime = executor->get_llvm_runtime();
  cached.persistent_ctx.arg_buffer = cached.persistent_device_arg_buffer;
  cached.persistent_ctx.result_buffer =
      (uint64 *)cached.persistent_device_result_buffer;
  cached.persistent_ctx.cpu_thread_id = 0;

  // --- Build CUDA graph ---
  void *graph = nullptr;
  CUDADriver::get_instance().graph_create(&graph, 0);

  void *prev_node = nullptr;
  for (const auto &task : offloaded_tasks) {
    void *ctx_ptr = &cached.persistent_ctx;
    prev_node = add_kernel_node(
        graph, prev_node, cuda_module->lookup_function(task.name),
        (unsigned int)task.grid_dim, (unsigned int)task.block_dim,
        (unsigned int)task.dynamic_shared_array_bytes, &ctx_ptr);
  }

  // --- Instantiate and launch ---
  CUDADriver::get_instance().graph_instantiate(&cached.graph_exec, graph,
                                               nullptr, nullptr, 0);

  auto *stream = CUDAContext::get_instance().get_stream();
  CUDADriver::get_instance().graph_launch(cached.graph_exec, stream);

  CUDADriver::get_instance().graph_destroy(graph);

  QD_TRACE("CUDA graph created with {} kernel nodes for launch_id={}",
           offloaded_tasks.size(), launch_id);

  cache_.emplace(launch_id, std::move(cached));
  used_on_last_call_ = true;
  return true;
}

}  // namespace cuda
}  // namespace quadrants::lang
