#include <map>

#include "quadrants/runtime/cuda/kernel_launcher.h"
#include "quadrants/runtime/cuda/cuda_utils.h"
#include "quadrants/rhi/cuda/cuda_context.h"
#include "quadrants/rhi/cuda/cuda_driver.h"
#include "quadrants/rhi/cuda/cuda_stream_pin.h"
#include "quadrants/runtime/llvm/llvm_runtime_executor.h"
#include "quadrants/program/graph_do_while_driver.h"
#include "quadrants/program/program.h"

#include <vector>

namespace quadrants::lang {
namespace cuda {

namespace {

constexpr std::size_t kAdStackMaxConcurrentThreads = 65536;

}  // namespace

void KernelLauncher::launch_offloaded_tasks(LaunchContextBuilder &ctx,
                                            JITModule *cuda_module,
                                            const std::vector<OffloadedTask> &offloaded_tasks,
                                            void *device_context_ptr) {
  auto *executor = get_runtime_executor();
  auto prepare_task = [&](std::size_t task_index, const OffloadedTask &task) -> int {
    int grid = task.grid_dim;
    if (task.ad_stack.per_thread_stride) {
      grid = std::max(1, std::min(grid, static_cast<int>(kAdStackMaxConcurrentThreads / task.block_dim)));
      executor->prepare_counted_adstack(task.ad_stack, std::size_t(grid) * task.block_dim, ctx);
    }
    return grid;
  };

  auto *active_stream = CUDAContext::get_instance().get_stream();
  for (size_t i = 0; i < offloaded_tasks.size();) {
    const auto &task = offloaded_tasks[i];
    if (task.stream_parallel_group_id == 0) {
      int effective_grid_dim = prepare_task(i, task);
      QD_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, effective_grid_dim, task.block_dim);
      cuda_module->launch(task.name, effective_grid_dim, task.block_dim, task.dynamic_shared_array_bytes,
                          {&ctx.get_context()}, {});
      i++;
    } else {
      size_t group_start = i;
      // Bound the fork/join batch to a single qd.graph_parallel_context() region/level/checkpoint via the shared
      // boundary helper (next_stream_parallel_run in llvm_compiled_data.h), the same definition
      // GraphManager::build_level uses. Two regions written back-to-back carry distinct graph_parallel_region_id and
      // emit no serial task between them to break the run, so without this the second region's sections would fork
      // alongside -- and race -- the first's, dropping the inter-region join (region B could read region A's writes
      // early). This streaming path is the pre-SM 9.0 fallback and every graph_do_while iteration that streams.
      i = next_stream_parallel_run(offloaded_tasks, i, offloaded_tasks.size());

      std::vector<int> grid_dims(i - group_start);
      for (size_t j = group_start; j < i; j++) {
        grid_dims[j - group_start] = prepare_task(j, offloaded_tasks[j]);
      }

      // Record an event on the default stream so pool streams can wait for the arg_buffer upload and any per-task
      // metadata copies (memcpy_host_to_device_async on `active_stream`) without stalling the CPU.  Pool streams are
      // CU_STREAM_NON_BLOCKING and do not implicitly synchronize with the default stream.
      void *upload_event = nullptr;
      CUDADriver::get_instance().event_create(&upload_event, 0x2 /*CU_EVENT_DISABLE_TIMING*/);
      CUDADriver::get_instance().event_record(upload_event, active_stream);

      std::map<int, void *> stream_by_id;
      for (size_t j = group_start; j < i; j++) {
        int sid = offloaded_tasks[j].stream_parallel_group_id;
        if (stream_by_id.find(sid) == stream_by_id.end()) {
          stream_by_id[sid] = CUDAContext::get_instance().acquire_stream();
        }
      }

      try {
        for (auto &[sid, s] : stream_by_id) {
          CUDADriver::get_instance().stream_wait_event(s, upload_event, 0);
        }
        for (size_t j = group_start; j < i; j++) {
          const auto &t = offloaded_tasks[j];
          CUDAContext::get_instance().set_stream(stream_by_id[t.stream_parallel_group_id]);
          QD_TRACE("Launching kernel {}<<<{}, {}>>>", t.name, grid_dims[j - group_start], t.block_dim);
          cuda_module->launch(t.name, grid_dims[j - group_start], t.block_dim, t.dynamic_shared_array_bytes,
                              {&ctx.get_context()}, {});
        }

        // Join: record an event on each pool stream and make the default stream wait, so subsequent serial work on
        // active_stream orders after the parallel group without stalling the CPU.
        for (auto &[sid, s] : stream_by_id) {
          void *done = nullptr;
          CUDADriver::get_instance().event_create(&done, 0x2 /*CU_EVENT_DISABLE_TIMING*/);
          CUDADriver::get_instance().event_record(done, s);
          CUDADriver::get_instance().stream_wait_event(active_stream, done, 0);
          CUDADriver::get_instance().event_destroy(done);
        }
      } catch (...) {
        for (auto &[sid, s] : stream_by_id) {
          CUDAContext::get_instance().release_stream(s);
        }
        CUDADriver::get_instance().event_destroy(upload_event);
        CUDAContext::get_instance().set_stream(active_stream);
        throw;
      }
      for (auto &[sid, s] : stream_by_id) {
        CUDAContext::get_instance().release_stream(s);
      }
      CUDADriver::get_instance().event_destroy(upload_event);

      CUDAContext::get_instance().set_stream(active_stream);
    }
  }
}

void KernelLauncher::launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx,
                                                          JITModule *cuda_module,
                                                          const std::vector<OffloadedTask> &offloaded_tasks,
                                                          void *device_context_ptr) {
  const auto &levels = ctx.graph_do_while_levels;
  auto read_flag = [&](int level) -> bool {
    int32_t counter_val = 0;
    auto *stream = CUDAContext::get_instance().get_stream();
    CUDADriver::get_instance().memcpy_device_to_host_async(&counter_val, levels[level].flag_dev_ptr, sizeof(int32_t),
                                                           stream);
    CUDADriver::get_instance().stream_synchronize(stream);
    return counter_val != 0;
  };

  std::vector<int> level_per_task(offloaded_tasks.size());
  for (size_t i = 0; i < offloaded_tasks.size(); i++) {
    level_per_task[i] = offloaded_tasks[i].graph_do_while_level_id;
  }
  const bool has_top_level_task =
      std::any_of(level_per_task.begin(), level_per_task.end(), [](int l) { return l < 0; });

  if (levels.size() == 1 && !has_top_level_task) {
    // Single loop with every task inside it: re-run the full task list (with stream-parallel groups / adstack) each
    // iteration. Kernels that mix the loop with plain top-level for-loops (which run once) fall through to the general
    // driver below.
    do {
      launch_offloaded_tasks(ctx, cuda_module, offloaded_tasks, device_context_ptr);
    } while (read_flag(0));
    return;
  }

  // Nested or for-loop-mixed graph_do_while host fallback (pre-SM 9.0 only; SM 9.0+ uses native conditional graph
  // nodes). Simple per-task dispatch driven by the per-task level tags; stream-parallel groups and reverse-mode adstack
  // are not combined with this path.
  auto launch_task = [&](int i) -> bool {
    const auto &task = offloaded_tasks[i];
    cuda_module->launch(task.name, task.grid_dim, task.block_dim, task.dynamic_shared_array_bytes, {&ctx.get_context()},
                        {});
    return true;
  };
  auto continue_level = [&](int level) -> bool { return read_flag(level); };
  run_graph_do_while((int)offloaded_tasks.size(), level_per_task, levels, launch_task, continue_level);
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

  auto &launcher_ctx = contexts_[handle.get_launch_id()];
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

  // Default-stream fast path: every HtoD / DtoH / kernel-dispatch in this launcher already routes through
  // `active_stream`, so when entry `stream_ == nullptr` AND every offloaded task launches on the same `active_stream`
  // (i.e. `stream_parallel_group_id == 0`), the entire chain serialises on the legacy default stream. The
  // `stream_synchronize` barriers between phases collapse to no-ops the surrounding sync DtoH (host-blocking on
  // pageable host memory) already drains, and the `CudaDefaultStreamPinGuard` re-pins `CUDAContext::stream_` to
  // nullptr defensively across the launch in case an inner helper temporarily swaps it. Outside the fast path -
  // user-supplied stream OR any task on `stream_parallel_group_id != 0` (per-group acquired streams differ from
  // `active_stream`) - the cross-stream barriers below are load-bearing for HtoD / kernel / DtoH visibility and the
  // pin guard would silently override the user-requested stream at the kernel-launch site, so the guard stays
  // disengaged and main's sync semantics remain untouched. Closes the cross-stream-visibility break that
  // `cuMemAllocAsync`-pool buffers show on pre-Ampere hardware (Turing T4 faults at `cuLaunchKernel` with
  // `illegal-address` when consumer + producer streams differ) on the default-stream path.
  const bool all_sgid_zero = std::all_of(offloaded_tasks.begin(), offloaded_tasks.end(),
                                         [](const OffloadedTask &t) { return t.stream_parallel_group_id == 0; });
  const bool default_stream_path = (active_stream == nullptr) && all_sgid_zero;
  CudaDefaultStreamPinGuard cuda_pin(/*engage=*/default_stream_path);

  char *device_result_buffer{nullptr};
  // Launcher-global persistent `result_buffer`. See `kernel_launcher.h` for why this one is shared across handles
  // (kernel writes + synchronous host readback before any other reader runs). `arg_buffer` and `runtime_context`
  // are per-handle (in `Context`) to avoid recursive snode-reader launches clobbering the parent's args.
  const std::size_t needed_result = std::max(ctx.result_buffer_size, sizeof(uint64));
  if (needed_result > persistent_result_buffer_capacity_) {
    if (persistent_result_buffer_dev_ptr_ != nullptr) {
      CUDADriver::get_instance().mem_free_async(persistent_result_buffer_dev_ptr_, nullptr);
    }
    const std::size_t new_cap = std::max(needed_result, 2 * persistent_result_buffer_capacity_);
    CUDADriver::get_instance().malloc_async(&persistent_result_buffer_dev_ptr_, new_cap, nullptr);
    persistent_result_buffer_capacity_ = new_cap;
  }
  device_result_buffer = static_cast<char *>(persistent_result_buffer_dev_ptr_);
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
        ctx.resolve_graph_do_while_flag(arg_id, device_ptrs[data_ptr_idx]);
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
        ctx.resolve_graph_do_while_flag(arg_id, device_ptrs[data_ptr_idx]);
      }
    }
  }
  // On the default-stream fast path the post-HtoD `stream_synchronize` is redundant: HtoD goes on the null stream and
  // the subsequent `cuda_module->launch` reads `CUDAContext::stream_` (pinned to nullptr) so the kernel dispatch
  // serialises with the HtoD by null-stream ordering. Outside the fast path the barrier remains load-bearing because
  // HtoD on `active_stream` is async and per-group launches read it from a different stream.
  if (transfers.size() > 0 && !default_stream_path) {
    CUDADriver::get_instance().stream_synchronize(active_stream);
  }
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    ctx.get_context().result_buffer = (uint64 *)device_result_buffer;
  }
  // When launching on an explicit stream (active_stream != nullptr), two calls to the same kernel on different streams
  // would race on the shared per-handle arg_buffer: the second call's memcpy can overwrite the buffer while the first
  // kernel is still reading it.  Allocate a per-call ephemeral buffer in that case; the stream-ordered free below
  // ensures the memory stays live until the kernel finishes.
  const bool use_persistent_scratch = (active_stream == nullptr);
  char *device_arg_buffer = nullptr;
  void *ephemeral_arg_buffer = nullptr;
  if (ctx.arg_buffer_size > 0) {
    if (use_persistent_scratch) {
      if (ctx.arg_buffer_size > launcher_ctx.arg_buffer_capacity) {
        if (launcher_ctx.arg_buffer_dev_ptr != nullptr) {
          CUDADriver::get_instance().mem_free_async(launcher_ctx.arg_buffer_dev_ptr, nullptr);
        }
        const std::size_t new_cap = std::max<std::size_t>(ctx.arg_buffer_size, 2 * launcher_ctx.arg_buffer_capacity);
        CUDADriver::get_instance().malloc_async(&launcher_ctx.arg_buffer_dev_ptr, new_cap, nullptr);
        launcher_ctx.arg_buffer_capacity = new_cap;
      }
      device_arg_buffer = static_cast<char *>(launcher_ctx.arg_buffer_dev_ptr);
    } else {
      CUDADriver::get_instance().malloc_async(&ephemeral_arg_buffer, ctx.arg_buffer_size, active_stream);
      device_arg_buffer = static_cast<char *>(ephemeral_arg_buffer);
    }
    CUDADriver::get_instance().memcpy_host_to_device_async(device_arg_buffer, ctx.get_context().arg_buffer,
                                                           ctx.arg_buffer_size, active_stream);
    ctx.get_context().arg_buffer = device_arg_buffer;
  }

  void *device_context_ptr = nullptr;
  void *ephemeral_context_ptr = nullptr;

  if (ctx.has_graph_do_while()) {
    for (const auto &level : ctx.graph_do_while_levels) {
      QD_ASSERT(level.flag_dev_ptr);
    }
    launch_offloaded_tasks_with_do_while(ctx, cuda_module, offloaded_tasks, device_context_ptr);
  } else {
    launch_offloaded_tasks(ctx, cuda_module, offloaded_tasks, device_context_ptr);
  }
  // Persistent scratch (default-stream path): no per-launch free for the per-handle `arg_buffer` / `runtime_context`
  // or the launcher-global `result_buffer`. All live until launcher destruction; the dtor handles the final
  // `mem_free_async`.  Ephemeral buffers (explicit-stream path) are freed below.
  if (ctx.result_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_device_to_host_async(host_result_buffer, device_result_buffer,
                                                           ctx.result_buffer_size, active_stream);
  }
  // Copy data back to host. On the default-stream fast path the kernel ran on the null stream, so a sync
  // `cuMemcpyDtoH` (host-blocking, on null stream) sees the kernel's writes without an explicit cross-stream barrier
  // and host-drains the prior async `memcpy_device_to_host_async(host_result_buffer, ...)` queued on the same null
  // stream - the explicit `stream_synchronize(nullptr)` calls collapse to no-ops. Outside the fast path the barriers
  // remain load-bearing for cross-stream visibility (per-group kernel writes vs `active_stream` DtoH).
  if (transfers.size() > 0) {
    if (!default_stream_path) {
      CUDADriver::get_instance().stream_synchronize(active_stream);
    }
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      auto &idx = itr->first;
      if (default_stream_path) {
        CUDADriver::get_instance().memcpy_device_to_host(itr->second.first, (void *)device_ptrs[idx],
                                                         ctx.array_runtime_sizes[idx.arg_id]);
      } else {
        CUDADriver::get_instance().memcpy_device_to_host_async(itr->second.first, (void *)device_ptrs[idx],
                                                               ctx.array_runtime_sizes[idx.arg_id], active_stream);
      }
    }
    if (!default_stream_path) {
      CUDADriver::get_instance().stream_synchronize(active_stream);
    }
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      executor->deallocate_memory_on_device(itr->second.second);
    }
  } else if (ctx.result_buffer_size > 0 && !default_stream_path) {
    CUDADriver::get_instance().stream_synchronize(active_stream);
  }
  // Free per-call ephemeral buffers (explicit-stream path).  The free is stream-ordered: it won't execute until all
  // preceding work on active_stream (including the kernel reads) has completed.
  if (ephemeral_arg_buffer != nullptr) {
    CUDADriver::get_instance().mem_free_async(ephemeral_arg_buffer, active_stream);
  }
  if (ephemeral_context_ptr != nullptr) {
    CUDADriver::get_instance().mem_free_async(ephemeral_context_ptr, active_stream);
  }
}

KernelLauncher::~KernelLauncher() {
  // Free per-handle and launcher-global persistent scratch. `mem_free_async` queues behind any in-flight kernel
  // reads on the default stream, so the bytes stay valid until the launcher actually goes away.
  for (auto &launcher_ctx : contexts_) {
    if (launcher_ctx.arg_buffer_dev_ptr != nullptr) {
      CUDADriver::get_instance().mem_free_async(launcher_ctx.arg_buffer_dev_ptr, nullptr);
    }
    if (launcher_ctx.runtime_context_dev_ptr != nullptr) {
      CUDADriver::get_instance().mem_free_async(launcher_ctx.runtime_context_dev_ptr, nullptr);
    }
  }
  if (persistent_result_buffer_dev_ptr_ != nullptr) {
    CUDADriver::get_instance().mem_free_async(persistent_result_buffer_dev_ptr_, nullptr);
  }
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
