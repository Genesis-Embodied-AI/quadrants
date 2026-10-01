# Compile shader helpers into the build tree.
find_program(QD_GLSLANG_EXECUTABLE NAMES glslangValidator glslang
  HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
if(NOT QD_GLSLANG_EXECUTABLE)
  message(FATAL_ERROR "Shader helpers require glslang. Install the Vulkan SDK or set QD_GLSLANG_EXECUTABLE.")
endif()
execute_process(COMMAND "${QD_GLSLANG_EXECUTABLE}" --help
  OUTPUT_VARIABLE GLSLANG_HELP RESULT_VARIABLE GLSLANG_HELP_RESULT)
if(NOT GLSLANG_HELP_RESULT EQUAL 0 OR NOT GLSLANG_HELP MATCHES "--no-link")
  message(FATAL_ERROR "Shader helpers require glslang with --no-link support (13.1 or newer).")
endif()
find_package(Python COMPONENTS Interpreter REQUIRED)
set(WORKGROUP_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/workgroup_spv.h")
add_custom_command(OUTPUT "${WORKGROUP_HEADER}"
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/generate_workgroup.py"
    --compiler "${QD_GLSLANG_EXECUTABLE}" --output "${WORKGROUP_HEADER}"
  DEPENDS shaders/workgroup.comp shaders/generate_workgroup.py "${QD_GLSLANG_EXECUTABLE}"
  COMMENT "Compiling GLSL shader helpers"
  VERBATIM)

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
    "${WORKGROUP_HEADER}"
    spirv_types.cpp
    compiled_kernel_data.cpp
    kernel_compiler.cpp
  )

target_include_directories(spirv_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
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
