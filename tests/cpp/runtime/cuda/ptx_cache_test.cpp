#include <filesystem>

#include "gtest/gtest.h"

#include "quadrants/runtime/cuda/ptx_cache.h"

namespace quadrants::lang {

TEST(PtxCache, TestBasic) {
  auto temp_dir = std::filesystem::temp_directory_path() / "PtxCache.TestBasic";

  if (!std::filesystem::create_directory(temp_dir)) {
    FAIL() << "Failed to create temporary directory";
  }
  auto cleanup = [temp_dir]() { std::filesystem::remove_all(temp_dir); };

  struct DirCleaner {
    std::function<void()> cleaner;
    ~DirCleaner() {
      cleaner();
    }
  } dir_cleaner{cleanup};

  PtxCache::Config config;
  config.offline_cache_path = temp_dir.string();

  CompileConfig compile_config;
  compile_config.arch = Arch::cuda;

  std::unique_ptr<PtxCache> ptx_cache =
      std::make_unique<PtxCache>(config, compile_config);

  std::string llvm_ir1 = "some ir code1";
  std::string llvm_ir2 = "some ir code2";
  std::string ptx_code1 = "ptx code for test kernel1";
  std::string ptx_code1fast = "ptx code for test kernel1fast";
  std::string ptx_code2 = "ptx code for test kernel2";

  std::string key_1 = ptx_cache->make_cache_key(llvm_ir1, false);
  std::string key_1fast = ptx_cache->make_cache_key(llvm_ir1, true);
  std::string key_2 = ptx_cache->make_cache_key(llvm_ir2, false);

  ASSERT_NE(key_1, key_2);
  ASSERT_NE(key_1, key_1fast);

  ASSERT_EQ(std::nullopt, ptx_cache->load_ptx(key_1));
  ASSERT_EQ(std::nullopt, ptx_cache->load_ptx(key_2));
  ASSERT_EQ(std::nullopt, ptx_cache->load_ptx(key_1fast));

  ptx_cache->store_ptx(key_1, ptx_code1);
  ptx_cache->dump();

  ptx_cache = std::make_unique<PtxCache>(config, compile_config);
  ASSERT_EQ(ptx_code1, ptx_cache->load_ptx(key_1));
  ASSERT_EQ(std::nullopt, ptx_cache->load_ptx(key_2));
  ASSERT_EQ(std::nullopt, ptx_cache->load_ptx(key_1fast));
  ptx_cache->store_ptx(key_1fast, ptx_code1fast);
  ptx_cache->dump();

  ptx_cache = std::make_unique<PtxCache>(config, compile_config);
  ASSERT_EQ(ptx_code1, ptx_cache->load_ptx(key_1));
  ASSERT_EQ(std::nullopt, ptx_cache->load_ptx(key_2));
  ASSERT_EQ(ptx_code1fast, ptx_cache->load_ptx(key_1fast));
  ptx_cache->store_ptx(key_2, ptx_code2);
  ptx_cache->dump();

  ptx_cache = std::make_unique<PtxCache>(config, compile_config);
  ASSERT_EQ(ptx_code1, ptx_cache->load_ptx(key_1));
  ASSERT_EQ(ptx_code2, ptx_cache->load_ptx(key_2));
  ASSERT_EQ(ptx_code1fast, ptx_cache->load_ptx(key_1fast));
}

}  // namespace quadrants::lang
