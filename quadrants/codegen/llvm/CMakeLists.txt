# ./quadrants/codegen/llvm/CMakeLists.txt

add_library(llvm_codegen)
target_sources(llvm_codegen
  PRIVATE
    codegen_llvm.cpp
    codegen_llvm_quant.cpp
    llvm_codegen_utils.cpp
    struct_llvm.cpp
    compiled_kernel_data.cpp
    kernel_compiler.cpp
  )

target_include_directories(llvm_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(llvm_codegen PRIVATE quadrants_util)
target_link_libraries(llvm_codegen PRIVATE llvm_runtime)
# `codegen_llvm.cpp` calls `quadrants::lang::spirv::compile_snode_structs` to resolve the SNode descriptor for
# captured `bound_expr` gating fields (the same descriptor the SPIR-V backend compiles). MSVC's linker requires
# the link dep to be explicit; clang / gcc on Linux / Mac happen to satisfy it transitively via the final
# executable / shared module link order.
target_link_libraries(llvm_codegen PRIVATE spirv_codegen)
