#include "quadrants/analysis/bls_analyzer.h"
#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/scratch_pad.h"
#include "quadrants/system/profiler.h"

namespace quadrants::lang {

// TODO: rename scratch_pad to block_local_cache? Need to get rid of the
// scratch_pad term

namespace irpass {

std::unique_ptr<ScratchPads> initialize_scratch_pad(OffloadedStmt *offload) {
  QD_AUTO_PROF
  QD_ASSERT(offload->task_type == OffloadedTaskType::struct_for);
  std::unique_ptr<ScratchPads> pads;
  pads = std::make_unique<ScratchPads>();
  for (auto snode : offload->mem_access_opt.get_snodes_with_flag(
           SNodeAccessFlag::block_local)) {
    pads->insert(snode);
  }
  BLSAnalyzer bls_analyzer(offload, pads.get());
  bool analysis_ok = bls_analyzer.run();
  if (!analysis_ok) {
    QD_ERROR("BLS analysis failed !");
  }
  pads->finalize();
  return pads;
}

}  // namespace irpass

}  // namespace quadrants::lang
