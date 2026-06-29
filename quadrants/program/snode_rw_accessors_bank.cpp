#include "quadrants/program/snode_rw_accessors_bank.h"

#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/program.h"

namespace quadrants::lang {

namespace {
void set_kernel_args(const std::vector<int> &I, int num_active_indices, LaunchContextBuilder *launch_ctx) {
  for (int i = 0; i < num_active_indices; i++) {
    launch_ctx->set_arg_int(i, I[i]);
  }
}
}  // namespace

SNodeRwAccessorsBank::Accessors SNodeRwAccessorsBank::get(SNode *snode) {
  auto &kernels = snode_to_kernels_[snode];
  if (kernels.reader == nullptr) {
    kernels.reader = &(program_->get_snode_reader(snode));
  }
  if (kernels.writer == nullptr) {
    kernels.writer = &(program_->get_snode_writer(snode));
  }
  return Accessors(snode, kernels, program_);
}

SNodeRwAccessorsBank::Accessors::Accessors(const SNode *snode, RwKernels &kernels, Program *prog)
    : snode_(snode), prog_(prog), reader_(kernels.reader), writer_(kernels.writer), kernels_(kernels) {
  QD_ASSERT(reader_ != nullptr);
  QD_ASSERT(writer_ != nullptr);
}

// Compile on first call, memoise the result in `slot`, and reuse on every subsequent call.
static const CompiledKernelData &get_or_compile(const CompiledKernelData *&slot, Program *prog, const Kernel &k) {
  if (slot == nullptr) {
    CompileResult compile_result = prog->compile_kernel(prog->compile_config(), prog->get_device_caps(), k);
    slot = &compile_result.compiled_kernel_data;
  }
  return *slot;
}

void SNodeRwAccessorsBank::Accessors::write_float(const std::vector<int> &I, float64 val) {
  auto launch_ctx = writer_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  launch_ctx.set_arg_float(snode_->num_active_indices, val);
  prog_->synchronize();
  const auto &compiled_kernel_data = get_or_compile(kernels_.writer_compiled, prog_, *writer_);
  prog_->launch_kernel(compiled_kernel_data, launch_ctx);
  // Drives invalidation of the SPIR-V per-task adstack metadata cache: a runtime adstack bound that
  // reads this snode (`SizeExpr::FieldLoad`) must see the GPU sizer re-run on the next launch after
  // any host-side mutation. Bumped per snode_id so the cache only evicts entries whose `size_expr`
  // actually depends on this specific snode.
  prog_->adstack_cache().bump_snode_write_gen(snode_->id);
}

float64 SNodeRwAccessorsBank::Accessors::read_float(const std::vector<int> &I) {
  prog_->synchronize();
  auto launch_ctx = reader_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  const auto &compiled_kernel_data = get_or_compile(kernels_.reader_compiled, prog_, *reader_);
  prog_->launch_kernel(compiled_kernel_data, launch_ctx);
  prog_->synchronize();
  return launch_ctx.get_struct_ret_float({0});
}

// for int32 and int64
void SNodeRwAccessorsBank::Accessors::write_int(const std::vector<int> &I, int64 val) {
  auto launch_ctx = writer_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  launch_ctx.set_arg_int(snode_->num_active_indices, val);
  prog_->synchronize();
  const auto &compiled_kernel_data = get_or_compile(kernels_.writer_compiled, prog_, *writer_);
  prog_->launch_kernel(compiled_kernel_data, launch_ctx);
  prog_->adstack_cache().bump_snode_write_gen(snode_->id);
}

// for int32 and int64
void SNodeRwAccessorsBank::Accessors::write_uint(const std::vector<int> &I, uint64 val) {
  auto launch_ctx = writer_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  launch_ctx.set_arg_uint(snode_->num_active_indices, val);
  prog_->synchronize();
  const auto &compiled_kernel_data = get_or_compile(kernels_.writer_compiled, prog_, *writer_);
  prog_->launch_kernel(compiled_kernel_data, launch_ctx);
  prog_->adstack_cache().bump_snode_write_gen(snode_->id);
}

int64 SNodeRwAccessorsBank::Accessors::read_int(const std::vector<int> &I) {
  prog_->synchronize();
  auto launch_ctx = reader_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  const auto &compiled_kernel_data = get_or_compile(kernels_.reader_compiled, prog_, *reader_);
  prog_->launch_kernel(compiled_kernel_data, launch_ctx);
  prog_->synchronize();
  return launch_ctx.get_struct_ret_int({0});
}

uint64 SNodeRwAccessorsBank::Accessors::read_uint(const std::vector<int> &I) {
  prog_->synchronize();
  auto launch_ctx = reader_->make_launch_context();
  set_kernel_args(I, snode_->num_active_indices, &launch_ctx);
  const auto &compiled_kernel_data = get_or_compile(kernels_.reader_compiled, prog_, *reader_);
  prog_->launch_kernel(compiled_kernel_data, launch_ctx);
  prog_->synchronize();
  return launch_ctx.get_struct_ret_uint({0});
}

}  // namespace quadrants::lang
