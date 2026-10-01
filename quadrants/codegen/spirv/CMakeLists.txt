# ./quadrants/codegen/spirv/CMakeLists.txt

# Embed the precompiled GLSL library so wheel builds do not require glslang.
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/shaders/workgroup.comp" WORKGROUP_SOURCE_SHA256)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/shaders/workgroup_spv.h" WORKGROUP_HEADER)
string(FIND "${WORKGROUP_HEADER}" "GLSL source SHA256: ${WORKGROUP_SOURCE_SHA256}" WORKGROUP_HASH_POSITION)
if(WORKGROUP_HASH_POSITION EQUAL -1)
  message(FATAL_ERROR "Regenerate shaders/workgroup_spv.h with shaders/generate_workgroup.py after editing workgroup.comp")
endif()

add_library(spirv_codegen)
target_sources(spirv_codegen
  PRIVATE
    adstack_bound_reducer_shader.cpp
    adstack_max_reducer_shader.cpp
    adstack_sizer_shader.cpp
    checkpoint_gate_shader.cpp
    checkpoint_yield_check_shader.cpp
    kernel_utils.cpp
    snode_struct_compiler.cpp
    spirv_codegen.cpp
    spirv_shared_array_retyping.cpp
    spirv_ir_builder.cpp
    shader_library.cpp
    spirv_types.cpp
    compiled_kernel_data.cpp
    kernel_compiler.cpp
  )

target_include_directories(spirv_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Headers/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Reflect
    ${PROJECT_SOURCE_DIR}/external/FP16/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(spirv_codegen PRIVATE quadrants_util)
# NOTE: SPIRV-Tools-opt must come before SPIRV-Tools
# https://github.com/KhronosGroup/SPIRV-Tools/issues/1569#issuecomment-390250792
target_link_libraries(spirv_codegen PRIVATE SPIRV-Tools-link SPIRV-Tools-opt ${SPIRV_TOOLS})
