#include "gtest/gtest.h"

#include "quadrants/program/compile_config.h"
#include "quadrants/runtime/cuda/jit_cuda.h"

namespace quadrants::lang {

TEST(JitCudaSessionNonce, OfflineCacheEnabledLeavesPtxUntouched) {
  CompileConfig compile_config;
  compile_config.arch = Arch::cuda;
  compile_config.offline_cache = true;

  std::string ptx = ".version 7.0\n.target sm_80\n";
  const std::string original = ptx;
  append_compute_cache_bypass_nonce_if_disabled(ptx, compile_config);
  EXPECT_EQ(original, ptx);
}

TEST(JitCudaSessionNonce, OfflineCacheDisabledAppendsStableNonce) {
  CompileConfig compile_config;
  compile_config.arch = Arch::cuda;
  compile_config.offline_cache = false;

  std::string ptx_a = ".version 7.0\n";
  std::string ptx_b = ".version 7.0\n";

  append_compute_cache_bypass_nonce_if_disabled(ptx_a, compile_config);
  append_compute_cache_bypass_nonce_if_disabled(ptx_b, compile_config);

  // Nonce was appended.
  EXPECT_GT(ptx_a.size(), std::string(".version 7.0\n").size());
  EXPECT_NE(ptx_a.find("quadrants-session-nonce"), std::string::npos);

  // Two calls within the same process must produce identical output: the nonce is process-stable so kernels with
  // identical PTX still hit the driver compute cache within one run. Cross-process reuse is broken instead, by the
  // nonce changing between processes.
  EXPECT_EQ(ptx_a, ptx_b);
}

}  // namespace quadrants::lang
