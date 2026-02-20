#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "quadrants/common/core.h"
#include "quadrants/compilation_manager/kernel_compilation_manager.h"
#include "quadrants/codegen/compiled_kernel_data.h"
#include "quadrants/codegen/kernel_compiler.h"
#include "quadrants/program/kernel.h"
#include "quadrants/program/program.h"
#include "quadrants/program/compile_config.h"
#include "quadrants/util/offline_cache.h"
#include "quadrants/ir/ir.h"

namespace quadrants::lang {
namespace tests {

static constexpr Arch kFakeArch = (Arch)1024;

class FakeCompiledKernelData : public CompiledKernelData {
 public:
  FakeCompiledKernelData(const std::string &data = "test_data") : data_(data) {
  }

  Arch arch() const override {
    return kFakeArch;
  }

  std::unique_ptr<CompiledKernelData> clone() const override {
    return std::make_unique<FakeCompiledKernelData>(data_);
  }

  const std::string &get_data() const {
    return data_;
  }

 protected:
  Err load_impl(const CompiledKernelDataFile &file) override {
    if (file.arch() != kFakeArch) {
      return CompiledKernelData::Err::kArchNotMatched;
    }
    data_ = file.src_code();
    return CompiledKernelData::Err::kNoError;
  }

  Err dump_impl(CompiledKernelDataFile &file) const override {
    file.set_arch(kFakeArch);
    file.set_metadata("{}");
    file.set_src_code(data_);
    return CompiledKernelData::Err::kNoError;
  }

 private:
  std::string data_;
};

class FakeKernelCompiler : public KernelCompiler {
 public:
  IRNodePtr compile(const CompileConfig &, const Kernel &) const override {
    return std::make_unique<Block>();
  }

  CKDPtr compile(const CompileConfig &,
                 const DeviceCapabilityConfig &,
                 const Kernel &,
                 IRNode &) const override {
    return std::make_unique<FakeCompiledKernelData>("compiled_data");
  }
};

class KernelCompilationManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    temp_dir_ = std::filesystem::temp_directory_path() /
                ("kcm_test_" + std::to_string(std::time(nullptr)));
    std::filesystem::create_directories(temp_dir_);

    KernelCompilationManager::Config config;
    config.offline_cache_path = temp_dir_.string();
    config.kernel_compiler = std::make_unique<FakeKernelCompiler>();
    mgr_ = std::make_unique<KernelCompilationManager>(std::move(config));
  }

  void TearDown() override {
    mgr_.reset();
    if (std::filesystem::exists(temp_dir_)) {
      std::filesystem::remove_all(temp_dir_);
    }
  }

  std::filesystem::path temp_dir_;
  std::unique_ptr<KernelCompilationManager> mgr_;
  CompileConfig compile_config_;
  DeviceCapabilityConfig device_caps_;
};

TEST_F(KernelCompilationManagerTest, DumpNewKernel) {
  compile_config_.offline_cache = true;
  Program prog(Arch::x64);
  Kernel kernel(prog, [] {}, "test_kernel", AutodiffMode::kNone);

  auto ckd = std::make_unique<FakeCompiledKernelData>("test_compiled_data");

  std::string checksum = "test_kernel_key_123";
  mgr_->cache_kernel(checksum, compile_config_, std::move(ckd), kernel);
  mgr_->dump();
  auto cache_file =
      temp_dir_ / "kernel_compilation_manager" / (checksum + ".qdc");
  EXPECT_TRUE(std::filesystem::exists(cache_file));
  auto metadata_file = temp_dir_ / "kernel_compilation_manager" / "qdcache.tcb";
  EXPECT_TRUE(std::filesystem::exists(metadata_file));
}

TEST_F(KernelCompilationManagerTest, CacheExistingKernelThrowsException) {
  compile_config_.offline_cache = true;
  Program prog(Arch::x64);
  Kernel kernel(prog, [] {}, "test_kernel", AutodiffMode::kNone);

  std::string checksum = "existing_kernel_key_456";
  {
    auto ckd1 = std::make_unique<FakeCompiledKernelData>("old_data");
    mgr_->cache_kernel(checksum, compile_config_, std::move(ckd1), kernel);
  }
  auto ckd2 = std::make_unique<FakeCompiledKernelData>("new_data");
  EXPECT_ANY_THROW(
      mgr_->cache_kernel(checksum, compile_config_, std::move(ckd2), kernel));
}

TEST_F(KernelCompilationManagerTest, DumpMemCacheOnlyKernel) {
  // Test that MemCache-only kernels are not written to disk
  compile_config_.offline_cache = false;  // Disable offline cache
  Program prog(Arch::x64);
  Kernel kernel(prog, [] {}, "mem_only_kernel", AutodiffMode::kNone);

  auto ckd = std::make_unique<FakeCompiledKernelData>("mem_data");

  std::string checksum = "mem_cache_key";
  mgr_->cache_kernel(checksum, compile_config_, std::move(ckd), kernel);

  mgr_->dump();

  // Verify the kernel data was NOT written to disk
  auto cache_file =
      temp_dir_ / "kernel_compilation_manager" / (checksum + ".qdc");
  EXPECT_FALSE(std::filesystem::exists(cache_file));
}

TEST_F(KernelCompilationManagerTest, DumpMultipleKernels) {
  compile_config_.offline_cache = true;
  Program prog(Arch::x64);
  Kernel kernel1(prog, [] {}, "kernel1", AutodiffMode::kNone);
  Kernel kernel2(prog, [] {}, "kernel2", AutodiffMode::kNone);

  auto ckd1 = std::make_unique<FakeCompiledKernelData>("data1");
  auto ckd2 = std::make_unique<FakeCompiledKernelData>("data2");

  mgr_->cache_kernel("key1", compile_config_, std::move(ckd1), kernel1);
  mgr_->cache_kernel("key2", compile_config_, std::move(ckd2), kernel2);

  mgr_->dump();

  auto cache_file1 = temp_dir_ / "kernel_compilation_manager" / "key1.qdc";
  auto cache_file2 = temp_dir_ / "kernel_compilation_manager" / "key2.qdc";
  EXPECT_TRUE(std::filesystem::exists(cache_file1));
  EXPECT_TRUE(std::filesystem::exists(cache_file2));
}

TEST_F(KernelCompilationManagerTest, DumpEmptyCache) {
  // Test that dumping an empty cache doesn't crash
  mgr_->dump();
  // Should complete without error
}

}  // namespace tests
}  // namespace quadrants::lang
