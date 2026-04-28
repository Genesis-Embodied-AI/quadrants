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
  compile_config.offline_cache = true;  // Exercises the disk-persistence path; mem-only path is covered separately.

  // Use a test compute capability value
  int compute_capability = 80;  // SM 8.0 for testing

  std::unique_ptr<PtxCache> ptx_cache = std::make_unique<PtxCache>(config, compile_config, compute_capability);

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

  ptx_cache = std::make_unique<PtxCache>(config, compile_config, compute_capability);
  ASSERT_EQ(ptx_code1, ptx_cache->load_ptx(key_1));
  ASSERT_EQ(std::nullopt, ptx_cache->load_ptx(key_2));
  ASSERT_EQ(std::nullopt, ptx_cache->load_ptx(key_1fast));
  ptx_cache->store_ptx(key_1fast, ptx_code1fast);
  ptx_cache->dump();

  ptx_cache = std::make_unique<PtxCache>(config, compile_config, compute_capability);
  ASSERT_EQ(ptx_code1, ptx_cache->load_ptx(key_1));
  ASSERT_EQ(std::nullopt, ptx_cache->load_ptx(key_2));
  ASSERT_EQ(ptx_code1fast, ptx_cache->load_ptx(key_1fast));
  ptx_cache->store_ptx(key_2, ptx_code2);
  ptx_cache->dump();

  ptx_cache = std::make_unique<PtxCache>(config, compile_config, compute_capability);
  ASSERT_EQ(ptx_code1, ptx_cache->load_ptx(key_1));
  ASSERT_EQ(ptx_code2, ptx_cache->load_ptx(key_2));
  ASSERT_EQ(ptx_code1fast, ptx_cache->load_ptx(key_1fast));
}

TEST(PtxCache, TestSmVersionPartitioning) {
  auto temp_dir = std::filesystem::temp_directory_path() / "PtxCache.TestSmVersionPartitioning";

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
  compile_config.offline_cache = true;  // Exercises the disk-persistence path; mem-only path is covered separately.

  // Test with SM 8.0
  int compute_capability_80 = 80;
  std::unique_ptr<PtxCache> ptx_cache_80 = std::make_unique<PtxCache>(config, compile_config, compute_capability_80);

  // Test with SM 7.5
  int compute_capability_75 = 75;
  std::unique_ptr<PtxCache> ptx_cache_75 = std::make_unique<PtxCache>(config, compile_config, compute_capability_75);

  std::string llvm_ir = "some ir code";
  std::string ptx_code_80 = "ptx code for sm_80";
  std::string ptx_code_75 = "ptx code for sm_75";

  // Cache keys should be different for different SM versions
  std::string key_80 = ptx_cache_80->make_cache_key(llvm_ir, false);
  std::string key_75 = ptx_cache_75->make_cache_key(llvm_ir, false);
  ASSERT_NE(key_80, key_75);

  // Store PTX for both SM versions
  ptx_cache_80->store_ptx(key_80, ptx_code_80);
  ptx_cache_75->store_ptx(key_75, ptx_code_75);

  // Verify that each cache only loads its own PTX
  ASSERT_EQ(ptx_code_80, ptx_cache_80->load_ptx(key_80));
  ASSERT_EQ(std::nullopt, ptx_cache_80->load_ptx(key_75));

  ASSERT_EQ(ptx_code_75, ptx_cache_75->load_ptx(key_75));
  ASSERT_EQ(std::nullopt, ptx_cache_75->load_ptx(key_80));

  // Dump both caches
  ptx_cache_80->dump();
  ptx_cache_75->dump();

  // Verify that separate cache directories are created
  ASSERT_TRUE(std::filesystem::exists(temp_dir / "ptx_cache_sm_80"));
  ASSERT_TRUE(std::filesystem::exists(temp_dir / "ptx_cache_sm_75"));

  // Reload and verify persistence
  ptx_cache_80 = std::make_unique<PtxCache>(config, compile_config, compute_capability_80);
  ptx_cache_75 = std::make_unique<PtxCache>(config, compile_config, compute_capability_75);

  ASSERT_EQ(ptx_code_80, ptx_cache_80->load_ptx(key_80));
  ASSERT_EQ(ptx_code_75, ptx_cache_75->load_ptx(key_75));
}

TEST(PtxCache, OfflineCacheDisabledSkipsDisk) {
  auto temp_dir = std::filesystem::temp_directory_path() / "PtxCache.OfflineCacheDisabledSkipsDisk";
  std::filesystem::remove_all(temp_dir);
  ASSERT_TRUE(std::filesystem::create_directory(temp_dir));
  struct DirCleaner {
    std::filesystem::path path;
    ~DirCleaner() {
      std::filesystem::remove_all(path);
    }
  } dir_cleaner{temp_dir};

  PtxCache::Config config;
  config.offline_cache_path = temp_dir.string();

  CompileConfig compile_config;
  compile_config.arch = Arch::cuda;
  compile_config.offline_cache = false;

  int compute_capability = 80;
  auto sm_dir = temp_dir / ("ptx_cache_sm_" + std::to_string(compute_capability));

  auto ptx_cache = std::make_unique<PtxCache>(config, compile_config, compute_capability);
  std::string key = ptx_cache->make_cache_key("ir", false);

  // In-memory caching still works within a single instance.
  ptx_cache->store_ptx(key, "ptx-payload");
  ASSERT_EQ("ptx-payload", ptx_cache->load_ptx(key));

  // dump() must not write anything to disk when the offline cache is disabled.
  ptx_cache->dump();
  EXPECT_FALSE(std::filesystem::exists(sm_dir))
      << "ptx_cache_sm_* directory must not be created when offline_cache=false";

  // A fresh instance pointed at the same temp dir must not observe the previous payload.
  ptx_cache = std::make_unique<PtxCache>(config, compile_config, compute_capability);
  EXPECT_EQ(std::nullopt, ptx_cache->load_ptx(key));
}

}  // namespace quadrants::lang
