#include "quadrants/program/adstack/replay_measurement.h"
#include "quadrants/program/ndarray.h"
#include "quadrants/program/program.h"

namespace quadrants::lang {

std::vector<int64_t> read_adstack_replay_measurement(Program *program, Ndarray *buffer) {
  const auto arch = program->compile_config().arch;
  QD_ERROR_IF(arch != Arch::x64 && arch != Arch::cuda, "Direct adstack measurement readback requires CPU or CUDA");
  QD_ERROR_IF(buffer->get_nelement() != 2 || buffer->dtype != PrimitiveType::i64,
              "Adstack measurement must contain two i64 values");
  // The count/reverse launch has already materialized pending work. Drain the program's stream before the
  // synchronous device readback, including when the CUDA runtime uses a non-default stream internally.
  program->synchronize();
  auto allocation = buffer->get_device_allocation();
  auto source = allocation.get_ptr();
  std::vector<int64_t> result(2);
  void *destination = result.data();
  size_t size = result.size() * sizeof(int64_t);
  QD_ASSERT(allocation.device->readback_data(&source, &destination, &size) == RhiResult::success);
  return result;
}

}  // namespace quadrants::lang
