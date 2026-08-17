#pragma once

#include "quadrants/rhi/arch.h"
#include "quadrants/util/lang_util.h"

namespace quadrants::lang {

struct CompileConfig {
  // All option fields (names, types, and defaults) are generated from
  // tools/config_codegen/schema.py -- the single source of truth for qd.init
  // options. Literal defaults appear here as in-class member initializers;
  // computed defaults (e.g. arch = host_arch()) are declared here and assigned
  // in the generated constructor fragment (see compile_config.cpp). Run
  // tools/config_codegen/generate.py to regenerate; CMake does this at
  // configure time. DO NOT hand-add option fields here.
#include "quadrants/program/compile_config.fields.generated.inc"

  CompileConfig();

  void fit();
};

extern QD_DLL_EXPORT CompileConfig default_compile_config;

}  // namespace quadrants::lang
