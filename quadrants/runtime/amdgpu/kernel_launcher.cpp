#include "quadrants/runtime/amdgpu/kernel_launcher.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/program/launch_context_builder.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace quadrants::lang {
namespace amdgpu {

namespace exp12_diag {
static const bool diag_enabled = []() {
  const char *flag = std::getenv("QD_EXP12_DIAG_ON");
  return flag != nullptr && flag[0] != '\0' && flag[0] != '0';
}();

struct KernelBranchCounts {
  std::atomic<long> launches{0};
  std::atomic<long> kNone_on_device{0};
  std::atomic<long> kNone_host_copy{0};
  std::atomic<long> kNdarray_passthrough{0};
  std::atomic<long> skip{0};
};
static std::mutex stats_mutex;
static std::unordered_map<std::string, std::unique_ptr<KernelBranchCounts>>
    stats;
static std::atomic<bool> atexit_registered{false};

static void flush_to_side_log() {
  const char *out_path = std::getenv("QD_EXP12_DIAG_OUT");
  if (!out_path) {
    out_path = "/tmp/exp12_diag.csv";
  }
  std::ofstream out(out_path);
  if (!out.is_open()) {
    return;
  }
  out << "kernel_name,launches,kNone_on_device,kNone_host_copy,"
      << "kNdarray_passthrough,skip\n";
  std::lock_guard<std::mutex> lock(stats_mutex);
  for (const auto &kv : stats) {
    const auto &c = *kv.second;
    out << '"' << kv.first << '"' << ',' << c.launches.load() << ','
        << c.kNone_on_device.load() << ',' << c.kNone_host_copy.load() << ','
        << c.kNdarray_passthrough.load() << ',' << c.skip.load() << '\n';
  }
}

static KernelBranchCounts &get_or_create(const std::string &name) {
  std::lock_guard<std::mutex> lock(stats_mutex);
  if (!atexit_registered.exchange(true)) {
    std::atexit(flush_to_side_log);
  }
  auto it = stats.find(name);
  if (it == stats.end()) {
    it = stats.emplace(name, std::make_unique<KernelBranchCounts>()).first;
  }
  return *it->second;
}
}  // namespace exp12_diag

void KernelLauncher::launch_offloaded_tasks(LaunchContextBuilder &ctx,
                                            Context &launcher_ctx) {
  constexpr int kRuntimeContextArgSize = sizeof(RuntimeContext);
  // Hoist arg vectors out of the per-task loop. Initializer-list
  // arguments to launch() construct fresh std::vector<...> on every call
  // (heap alloc + free).
  std::vector<void *> arg_ptrs(1);
  const std::vector<int> arg_sizes{kRuntimeContextArgSize};
  arg_ptrs[0] = &ctx.get_context();
  const auto &offloaded_tasks = launcher_ctx.offloaded_tasks;
  auto &resolved = launcher_ctx.resolved_funcs;
  if (resolved.size() != offloaded_tasks.size()) {
    resolved.assign(offloaded_tasks.size(), nullptr);
  }
  auto *amdgpu_module = launcher_ctx.jit_module;
  for (size_t i = 0; i < offloaded_tasks.size(); ++i) {
    const auto &task = offloaded_tasks[i];
    QD_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
             task.block_dim);
    void *func = resolved[i];
    if (!func) {
      func = amdgpu_module->lookup_function(task.name);
      resolved[i] = func;
    }
    AMDGPUContext::get_instance().launch(func, task.name, arg_ptrs, arg_sizes,
                                         task.grid_dim, task.block_dim,
                                         task.dynamic_shared_array_bytes);
  }
}

void KernelLauncher::launch_offloaded_tasks_with_do_while(
    LaunchContextBuilder &ctx, Context &launcher_ctx) {
  int32_t counter_val;
  do {
    launch_offloaded_tasks(ctx, launcher_ctx);
    counter_val = 0;
    AMDGPUDriver::get_instance().stream_synchronize(nullptr);
    AMDGPUDriver::get_instance().memcpy_device_to_host(
        &counter_val, ctx.graph_do_while_flag_dev_ptr, sizeof(int32_t));
  } while (counter_val != 0);
}

bool KernelLauncher::on_amdgpu_device(void *ptr) {
  unsigned int attr_val[8];
  // mem_get_attribute doesn't work well on ROCm
  uint32_t ret_code =
      AMDGPUDriver::get_instance().mem_get_attributes.call(attr_val, ptr);

  return ret_code == HIP_SUCCESS && attr_val[0] == HIP_MEMORYTYPE_DEVICE;
}

void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
  QD_ASSERT(handle.get_launch_id() < contexts_.size());
  // Take by reference, not by value. The Context contains a
  // std::vector<OffloadedTask>; copying it on every kernel launch costs
  // a heap alloc, a copy, and a free. We need a non-const reference so
  // we can lazily populate launcher_ctx.resolved_funcs on first use.
  auto &launcher_ctx = contexts_[handle.get_launch_id()];
  auto *executor = get_runtime_executor();
  const auto &parameters = *launcher_ctx.parameters;
  const auto &offloaded_tasks = launcher_ctx.offloaded_tasks;

  // Hoist context_set_current out of the per-launch path. The HIP
  // driver setter is a global (locked) call that does real work even when
  // the current context is unchanged.
  {
    thread_local void *cached_set_ctx = nullptr;
    void *want_ctx = AMDGPUContext::get_instance().get_context();
    if (cached_set_ctx != want_ctx) {
      AMDGPUContext::get_instance().make_current();
      cached_set_ctx = want_ctx;
    }
  }
  ctx.get_context().runtime = executor->get_llvm_runtime();

  exp12_diag::KernelBranchCounts *branch_counts = nullptr;
  if (exp12_diag::diag_enabled && !offloaded_tasks.empty()) {
    branch_counts = &exp12_diag::get_or_create(offloaded_tasks.front().name);
    branch_counts->launches.fetch_add(1, std::memory_order_relaxed);
  }

  // We only construct the map when we actually have to track
  // a host<->device copy (kNone branch with host pointer).
  std::unique_ptr<
      std::unordered_map<ArgArrayPtrKey, std::pair<void *, DeviceAllocation>,
                         ArgArrayPtrKeyHasher>>
      transfers;
  // device_ptrs is only needed to keep the transfer entries' destination
  // pointer alive across the kernel call; in the passthrough path the
  // pointer is set into ctx and immediately consumed by the kernel, so
  // the map is unnecessary. We construct it lazily (same trigger as
  // `transfers`) so we don't pay the allocation cost on the hot path.
  std::unique_ptr<
      std::unordered_map<ArgArrayPtrKey, void *, ArgArrayPtrKeyHasher>>
      device_ptrs;

  // Only allocate the device result buffer when something actually needs it.
  bool needs_result_buffer = ctx.result_buffer_size > 0;
  if (!needs_result_buffer) {
    for (int i = 0; i < (int)parameters.size(); i++) {
      const auto &parameter = parameters[i].second;
      if (!parameter.is_array)
        continue;
      const auto arg_id = parameters[i].first;
      if (ctx.array_runtime_sizes[arg_id] == 0)
        continue;
      if (ctx.device_allocation_type[arg_id] ==
              LaunchContextBuilder::DevAllocType::kNone &&
          !on_amdgpu_device(
              ctx.array_ptrs[ArgArrayPtrKey{
                  arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY}])) {
        needs_result_buffer = true;
        break;
      }
    }
  }

  char *device_result_buffer{nullptr};
  if (needs_result_buffer) {
    AMDGPUDriver::get_instance().malloc_async(
        (void **)&device_result_buffer,
        std::max(ctx.result_buffer_size, sizeof(uint64)), nullptr);
  }

  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &arg_id = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[arg_id];
      if (arr_sz == 0) {
        if (branch_counts) {
          branch_counts->skip.fetch_add(1, std::memory_order_relaxed);
        }
        continue;
      }

      ArgArrayPtrKey data_ptr_idx{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      ArgArrayPtrKey grad_ptr_idx{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
      auto data_ptr = ctx.array_ptrs[data_ptr_idx];

      void *resolved_dev_ptr = nullptr;
      if (ctx.device_allocation_type[arg_id] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        if (on_amdgpu_device(data_ptr)) {
          if (branch_counts) {
            branch_counts->kNone_on_device.fetch_add(
                1, std::memory_order_relaxed);
          }
          resolved_dev_ptr = data_ptr;
        } else {
          if (branch_counts) {
            branch_counts->kNone_host_copy.fetch_add(
                1, std::memory_order_relaxed);
          }
          DeviceAllocation devalloc = executor->allocate_memory_on_device(
              arr_sz, (uint64 *)device_result_buffer);
          resolved_dev_ptr = executor->get_device_alloc_info_ptr(devalloc);
          if (!transfers) {
            transfers = std::make_unique<std::unordered_map<
                ArgArrayPtrKey, std::pair<void *, DeviceAllocation>,
                ArgArrayPtrKeyHasher>>();
            device_ptrs = std::make_unique<std::unordered_map<
                ArgArrayPtrKey, void *, ArgArrayPtrKeyHasher>>();
          }
          (*device_ptrs)[data_ptr_idx] = resolved_dev_ptr;
          (*transfers)[data_ptr_idx] = {data_ptr, devalloc};

          AMDGPUDriver::get_instance().memcpy_host_to_device(
              resolved_dev_ptr, data_ptr, arr_sz);
        }
        ctx.set_ndarray_ptrs(arg_id, (uint64)resolved_dev_ptr,
                             (uint64)ctx.array_ptrs[grad_ptr_idx]);
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = resolved_dev_ptr;
        }
      } else if (arr_sz > 0) {  // why use arr_sz constrain?
        if (branch_counts) {
          branch_counts->kNdarray_passthrough.fetch_add(
              1, std::memory_order_relaxed);
        }
        // Ndarray
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        // Unwrapped raw ptr on device
        resolved_dev_ptr = executor->get_device_alloc_info_ptr(*ptr);

        ctx.set_ndarray_ptrs(arg_id, (uint64)resolved_dev_ptr,
                             (uint64)ctx.array_ptrs[grad_ptr_idx]);
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = resolved_dev_ptr;
        }
      }
    }
  }
  if (transfers && !transfers->empty()) {
    AMDGPUDriver::get_instance().stream_synchronize(nullptr);
  }
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    // Malloc_Async and Free_Async are available after ROCm 5.4
    ctx.get_context().result_buffer = (uint64 *)device_result_buffer;
  }
  // Persistent thread-local device arg buffer. The arg buffer
  // contents change every launch but the *backing storage* doesn't need
  // to be reallocated on the GPU. Stream ordering on the default stream
  // (H2D queued before launch, next H2D queued after this launch) makes
  // it safe to reuse the same device buffer for back-to-back launches —
  // the next H2D won't overwrite anything the previous kernel still needs.
  //
  // This eliminates a malloc_async + mem_free_async pair on every kernel
  // launch.
  thread_local char *persistent_dev_arg_buf = nullptr;
  thread_local std::size_t persistent_dev_arg_buf_cap = 0;
  if (ctx.arg_buffer_size > 0) {
    if (ctx.arg_buffer_size > persistent_dev_arg_buf_cap) {
      if (persistent_dev_arg_buf) {
        AMDGPUDriver::get_instance().mem_free_async(persistent_dev_arg_buf,
                                                    nullptr);
      }
      // Round up to amortize future growth.
      std::size_t new_cap = std::max<std::size_t>(ctx.arg_buffer_size, 256);
      while (new_cap < ctx.arg_buffer_size) {
        new_cap *= 2;
      }
      AMDGPUDriver::get_instance().malloc_async(
          (void **)&persistent_dev_arg_buf, new_cap, nullptr);
      persistent_dev_arg_buf_cap = new_cap;
    }
    AMDGPUDriver::get_instance().memcpy_host_to_device_async(
        persistent_dev_arg_buf, ctx.get_context().arg_buffer,
        ctx.arg_buffer_size, nullptr);
    ctx.get_context().arg_buffer = persistent_dev_arg_buf;
  }

  if (ctx.graph_do_while_arg_id >= 0) {
    QD_ASSERT(ctx.graph_do_while_flag_dev_ptr);
    launch_offloaded_tasks_with_do_while(ctx, launcher_ctx);
  } else {
    launch_offloaded_tasks(ctx, launcher_ctx);
  }
  QD_TRACE("Launching kernel");
  if (ctx.result_buffer_size > 0) {
    // Async D2H so we don't force a host stall after every kernel.
    // The host_result_buffer is consumed later by the caller, which is
    // expected to synchronize when it actually needs the value.
    AMDGPUDriver::get_instance().memcpy_device_to_host_async(
        host_result_buffer, device_result_buffer, ctx.result_buffer_size,
        nullptr);
  }
  if (device_result_buffer) {
    AMDGPUDriver::get_instance().mem_free_async(device_result_buffer, nullptr);
  }
  if (transfers && !transfers->empty()) {
    // External-array round-trip path: we must wait for the kernel and the
    // async D2H above before reading the data back to the host buffers.
    AMDGPUDriver::get_instance().stream_synchronize(nullptr);
    for (auto itr = transfers->begin(); itr != transfers->end(); itr++) {
      auto &idx = itr->first;
      auto arg_id = idx.arg_id;
      AMDGPUDriver::get_instance().memcpy_device_to_host(
          itr->second.first, (void *)(*device_ptrs)[idx],
          ctx.array_runtime_sizes[arg_id]);
      executor->deallocate_memory_on_device(itr->second.second);
    }
  }
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(
    const LLVM::CompiledKernelData &compiled) {
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
