# ./quadrants/runtime/llvm/runtime_module/CMakeLists.txt

function(COMPILE_LLVM_RUNTIME rtm_arch)
    message(STATUS "Compiling LLVM byte code file for arch ${rtm_arch}")
    # The runtime is split across two translation units:
    #   - runtime.cpp: core runtime (allocator, NodeManager, range / mesh task helpers, printf, atomics, etc.)
    #   - adstack_runtime.cpp: adstack-specific helpers (per-launch device-resident SizeExpr / max-reducer / bound
    #     reducer, stack push / pop / top primitives, runtime_get_*adstack* getters, and the STRUCT_FIELD getters
    #     for the `adstack_*` LLVMRuntime fields)
    # Each TU is compiled to its own .bc, then llvm-link merges them into the final runtime_<arch>.bc that
    # `init_runtime_module` loads at host startup. Both TUs share the `LLVMRuntime` struct memory layout via
    # `llvm_runtime.h` so the field offsets line up across the link.
    #
    # -mno-outline-atomics: the merged bitcode is later JIT-compiled by the ORC JIT linker at runtime. On AArch64,
    # clang's default -moutline-atomics emits calls to helpers like __aarch64_ldadd4_acq_rel (normally provided by
    # libgcc/compiler-rt), but the ORC JIT environment has no access to those libraries and fails with "Symbols not
    # found". This flag forces clang to emit inline LL/SC atomic sequences directly in the bitcode instead. It is a
    # no-op on x86 and irrelevant for CUDA/GPU paths. The alternative (registering those symbols with the JIT linker)
    # would be far more complex and fragile.
    set(_clang_flags ${CLANG_OSX_FLAGS} -fno-exceptions -emit-llvm -std=c++17 -mno-outline-atomics -D "ARCH_${rtm_arch}" -I ${PROJECT_SOURCE_DIR})
    add_custom_target(
        "generate_llvm_runtime_${rtm_arch}"
        COMMAND ${CLANG_EXECUTABLE} ${_clang_flags} -c runtime.cpp -o "runtime_main_${rtm_arch}.bc"
        COMMAND ${CLANG_EXECUTABLE} ${_clang_flags} -c adstack_runtime.cpp -o "adstack_runtime_${rtm_arch}.bc"
        COMMAND ${LLVM_LINK_EXECUTABLE} "runtime_main_${rtm_arch}.bc" "adstack_runtime_${rtm_arch}.bc" -o "runtime_${rtm_arch}.bc"
        # TODO, it's better to avoid polluting the source dir, keep in build
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_dependencies(${CORE_LIBRARY_NAME} "generate_llvm_runtime_${rtm_arch}")
    install(FILES "${CMAKE_SOURCE_DIR}/quadrants/runtime/llvm/runtime_module/runtime_${arch}.bc" DESTINATION ${CMAKE_INSTALL_PREFIX}/python/quadrants/_lib/runtime)

endfunction()

function(COMPILE_CUSTOM_CUDA_LIBRARY cuda_path)
    add_custom_target(
        "generate_cuda_library"
        COMMAND ${CLANG_EXECUTABLE} -c cuda_runtime.cu --cuda-gpu-arch=sm_60 -emit-llvm -std=c++17 --no-cuda-version-check -nocudalib --cuda-path=${cuda_path}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_dependencies(${CORE_LIBRARY_NAME} "generate_cuda_library")
    install(FILES "${CMAKE_SOURCE_DIR}/quadrants/runtime/llvm/runtime_module/cuda_runtime-cuda-nvptx64-nvidia-cuda-sm_60.bc" DESTINATION ${CMAKE_INSTALL_PREFIX}/python/quadrants/_lib/runtime)

endfunction()

# Build llvm-runtime for host arch and cuda (if available)
foreach(arch IN LISTS HOST_ARCH CUDA_ARCH AMDGPU_ARCH)
  compile_llvm_runtime(${arch})
endforeach()

# You can implement hand-written CUDA Kernels in cuda_runtime.cu and then
# uncomment the following code to compile it into .bc files

#if(CUDA_ARCH)
#    find_package(CUDAToolkit)
#    if(${CUDAToolkit_FOUND})
#      if(${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 8)
#          compile_custom_cuda_library(${CUDAToolkit_TARGET_DIR})
#      endif()
#    endif()
#endif()
