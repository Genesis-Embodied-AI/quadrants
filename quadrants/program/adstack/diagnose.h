#pragma once

#include <cstdint>

#include "quadrants/ir/adstack_size_expr.h"

namespace quadrants::lang {

class Program;

// Diagnose-time variant that evaluates the same `SerializedSizeExpr` against the captured
// `AdStackCache::DiagnoseLaunchSnapshot` rather than a live `LaunchContextBuilder`. Used by
// `AdStackCache::diagnose_adstack_overflow` to resolve `ExternalTensorRead` / `ExternalTensorShape` leaves at
// error time against the live (potentially mutated) ndarray contents, without needing the launch ctx that is
// gone by sync time on async backends. The cross-backend `Device::map(*allocation, &host_ptr)` path is the
// design pivot - see `AdStackCache::DiagnoseLaunchSnapshot`'s comment for the rationale (vs. re-dispatching
// the on-device sizer). Returns -1 if any leaf cannot be resolved (e.g. an arg_id missing from the snapshot,
// or an allocation whose `Device::map` fails); callers fall back to the static dual-cause body in that case.
int64_t evaluate_adstack_size_expr_for_diagnose(const SerializedSizeExpr &expr, Program *prog);

}  // namespace quadrants::lang
