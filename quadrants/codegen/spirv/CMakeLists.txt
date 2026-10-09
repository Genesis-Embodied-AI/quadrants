# Compile shader helpers into the build tree.
find_program(QD_GLSLANG_EXECUTABLE NAMES glslangValidator glslang
  HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
if(NOT QD_GLSLANG_EXECUTABLE)
  message(FATAL_ERROR "Shader helpers require glslang. Install the Vulkan SDK or set QD_GLSLANG_EXECUTABLE.")
endif()
execute_process(COMMAND "${QD_GLSLANG_EXECUTABLE}" --version
  RESULT_VARIABLE GLSLANG_VERSION_RESULT OUTPUT_QUIET ERROR_VARIABLE GLSLANG_VERSION_ERROR)
if(NOT GLSLANG_VERSION_RESULT EQUAL 0)
  message(FATAL_ERROR "Cannot run glslang at \"${QD_GLSLANG_EXECUTABLE}\": ${GLSLANG_VERSION_RESULT} ${GLSLANG_VERSION_ERROR}")
endif()
execute_process(COMMAND "${QD_GLSLANG_EXECUTABLE}" --help
  OUTPUT_VARIABLE GLSLANG_HELP)
if(NOT GLSLANG_HELP MATCHES "--no-link")
  message(FATAL_ERROR "Shader helpers require glslang with --no-link support (13.1 or newer).")
endif()
# Compile a GLSL helper in input_filepath into a C++ array called array_name
# in the generated header file output_filepath. Make it available to build_target.
function(qd_add_shader_helper build_target input_filepath output_filepath array_name)
  set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/${input_filepath}")
  set(output_path "${CMAKE_CURRENT_BINARY_DIR}/generated/${output_filepath}")
  get_filename_component(output_directory "${output_path}" DIRECTORY)
  add_custom_command(OUTPUT "${output_path}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_directory}"
    # -V: Generate SPIR-V for Vulkan.
    # --target-env vulkan1.0: Target Vulkan 1.0 and SPIR-V 1.0.
    # We chose 1.0 because the helper only reads a workgroup index and needs no newer features. This lets it link with
    # our oldest supported kernel modules without raising their SPIR-V version.
    # --no-link: Compile a helper library without linking an entry point.
    # -Od: Disable optimization during shader compilation.
    # --vn: Write a C header containing an array with the specified name.
    # -o: Write the generated header to the specified output path.
    COMMAND "${QD_GLSLANG_EXECUTABLE}" -V --target-env vulkan1.0 --no-link -Od
      --vn "${array_name}" "${source_path}" -o "${output_path}"
    DEPENDS "${source_path}" "${QD_GLSLANG_EXECUTABLE}"
    COMMENT "Compiling GLSL helper ${input_filepath}"
    # Preserve each command argument, including paths containing spaces, as a single argument.
    VERBATIM)
  target_sources(${build_target} PRIVATE "${output_path}")
  target_include_directories(${build_target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
endfunction()

add_library(spirv_codegen)
qd_add_shader_helper(spirv_codegen shaders/workgroup.comp workgroup_spv.h workgroup_helper_spv)
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
    spirv_operations.cpp
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
