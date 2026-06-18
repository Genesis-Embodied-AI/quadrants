#include "quadrants/runtime/gfx/kernel_launcher.h"
#include "quadrants/codegen/spirv/compiled_kernel_data.h"
#include "quadrants/program/graph_do_while_driver.h"

namespace quadrants::lang {
namespace gfx {

KernelLauncher::KernelLauncher(Config config) : config_(std::move(config)) {
}

int32_t KernelLauncher::readback_graph_do_while_flag(LaunchContextBuilder &ctx, int cond_arg_id) {
  const ArgArrayPtrKey key{cond_arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
  auto it = ctx.array_ptrs.find(key);
  QD_ASSERT(it != ctx.array_ptrs.end());

  auto *device = config_.gfx_runtime_->get_ti_device();
  DeviceAllocation alloc = *(static_cast<DeviceAllocation *>(it->second));
  DevicePtr dev_ptr = alloc.get_ptr(0);

  int32_t flag_val = 0;
  void *host_ptr = &flag_val;
  size_t sz = sizeof(int32_t);
  QD_ASSERT(device->readback_data(&dev_ptr, &host_ptr, &sz, 1) == RhiResult::success);
  return flag_val;
}

void KernelLauncher::launch_offloaded_tasks_with_do_while(Handle handle, LaunchContextBuilder &ctx) {
  const auto &levels = ctx.graph_do_while_levels;

  // Build the per-task innermost level table from the SPIR-V TaskAttributes tags (mirrors the LLVM OffloadedTask
  // path). A task tagged -1 is a plain top-level task that runs exactly once.
  const int num_tasks = config_.gfx_runtime_->get_num_tasks(handle);
  std::vector<int> task_level_ids(num_tasks);
  bool has_top_level_task = false;
  for (int i = 0; i < num_tasks; ++i) {
    task_level_ids[i] = config_.gfx_runtime_->get_task_graph_do_while_level_id(handle, i);
    if (task_level_ids[i] < 0) {
      has_top_level_task = true;
    }
  }

  // Checkpoint gating mirrors CPU / AMDGPU: a yield (surfaced via the runtime's
  // last_yield_cp_id_on_last_call after a flush+sync) must exit the loop, otherwise the body re-enters,
  // skips every checkpoint, never decrements the user's counter, and spins forever. `from_checkpoint=cp`
  // applies only to the first pass, so we clear `ctx.resume_from_checkpoint` once a body has run.
  auto did_yield = [&]() -> bool { return config_.gfx_runtime_->last_yield_cp_id_on_last_call() != -1; };

  if (levels.size() == 1 && !has_top_level_task) {
    // Single loop whose body is the entire kernel: keep the historical fast path of recording every task in one
    // command list each iteration, which is materially cheaper than the per-task replay the general driver below
    // uses (one cmdlist + one args-buffer blit per iteration instead of per task).
    int32_t flag_val;
    do {
      config_.gfx_runtime_->launch_kernel(handle, ctx);
      config_.gfx_runtime_->synchronize();
      if (did_yield()) {
        break;
      }
      ctx.resume_from_checkpoint = -1;
      flag_val = readback_graph_do_while_flag(ctx, levels[0].cond_arg_id);
    } while (flag_val != 0);
    return;
  }

  // Nested / sibling loops, or a loop mixed with plain top-level for-loops: drive the loop tree on the host from
  // the per-task level tags. Each `launch_task` records exactly one offloaded task into the current command list;
  // `continue_level` flushes + waits so the just-recorded body's device writes are visible, checks for a yield
  // (exiting this and, by propagation, every enclosing loop), then reads the level's condition flag. GFX has no
  // device-side assert-abort hook here, so `launch_task` always reports success.
  auto launch_task = [&](int i) -> bool {
    config_.gfx_runtime_->launch_kernel(handle, ctx, i, i + 1);
    return true;
  };
  auto continue_level = [&](int level) -> bool {
    config_.gfx_runtime_->synchronize();
    if (did_yield()) {
      return false;
    }
    ctx.resume_from_checkpoint = -1;
    return readback_graph_do_while_flag(ctx, levels[level].cond_arg_id) != 0;
  };
  run_graph_do_while(num_tasks, task_level_ids, levels, launch_task, continue_level);
}

void KernelLauncher::launch_kernel(const lang::CompiledKernelData &compiled_kernel_data, LaunchContextBuilder &ctx) {
  auto handle = register_kernel(compiled_kernel_data);

  if (ctx.has_graph_do_while()) {
    launch_offloaded_tasks_with_do_while(handle, ctx);
  } else {
    config_.gfx_runtime_->launch_kernel(handle, ctx);
  }
}

KernelLauncher::Handle KernelLauncher::register_kernel(const lang::CompiledKernelData &compiled_kernel_data) {
  if (!compiled_kernel_data.get_handle()) {
    const auto *spirv_compiled = dynamic_cast<const spirv::CompiledKernelData *>(&compiled_kernel_data);
    const auto &spirv_data = spirv_compiled->get_internal_data();
    gfx::GfxRuntime::RegisterParams params;
    params.kernel_attribs = spirv_data.metadata.kernel_attribs;
    params.task_spirv_source_codes = spirv_data.src.spirv_src;
    params.num_snode_trees = spirv_data.metadata.num_snode_trees;
    auto h = config_.gfx_runtime_->register_quadrants_kernel(std::move(params));
    compiled_kernel_data.set_handle(h);
  }
  return *compiled_kernel_data.get_handle();
}

}  // namespace gfx
}  // namespace quadrants::lang
