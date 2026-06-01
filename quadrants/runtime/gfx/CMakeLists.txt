# ./quadrants/runtime/gfx/CMakeLists.txt

add_library(gfx_runtime)
target_sources(gfx_runtime
  PRIVATE
    runtime.cpp
    adstack_bound_reducer_launch.cpp
    adstack_max_reducer_launch.cpp
    adstack_sizer_launch.cpp
    snode_tree_manager.cpp
    kernel_launcher.cpp
  )
#TODO 4832, some dependencies here should not be required as they
# are build requirements of other targets.
target_include_directories(gfx_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/FP16/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
        ${LLVM_INCLUDE_DIRS}
  )
target_link_libraries(gfx_runtime PRIVATE quadrants_common)
# The SPIR-V adstack sizer is compiled from source at launch time via `spirv::build_adstack_sizer_spirv`
# (called from `GfxRuntime::launch_kernel`'s adstack metadata prep); link `spirv_codegen` here so the
# shader builder's symbol resolves on Linux where the host linker does not pull it in transitively.
target_link_libraries(gfx_runtime PRIVATE spirv_codegen)
