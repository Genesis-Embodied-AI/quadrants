#pragma once

// Umbrella header for the adstack subsystem. The implementation was split across
// `quadrants/program/adstack/{cache,eval,max_reducer,device_bytecode,diagnose,write_gen}.{cpp,h}`; this
// header re-includes the per-stage headers so existing call sites (`#include
// "quadrants/program/adstack_size_expr_eval.h"`) compile unchanged. New call sites should prefer
// including the specific stage header they depend on.

#include "quadrants/program/adstack/cache.h"
#include "quadrants/program/adstack/device_bytecode.h"
#include "quadrants/program/adstack/diagnose.h"
#include "quadrants/program/adstack/eval.h"
#include "quadrants/program/adstack/max_reducer.h"
#include "quadrants/program/adstack/write_gen.h"
