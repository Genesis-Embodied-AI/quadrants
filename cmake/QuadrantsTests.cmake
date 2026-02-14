cmake_minimum_required(VERSION 3.11)

set(TESTS_NAME quadrants_cpp_tests)
if (WIN32)
    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

# TODO(#2195):
# 1. "cpp" -> "cpp_legacy", "cpp_new" -> "cpp"
# 2. Re-implement the legacy CPP tests using googletest
file(GLOB_RECURSE QUADRANTS_TESTS_SOURCE
        "tests/cpp/analysis/*.cpp"
        "tests/cpp/backends/*.cpp"
        "tests/cpp/codegen/*.cpp"
        "tests/cpp/common/*.cpp"
        "tests/cpp/ir/*.cpp"
        "tests/cpp/compilation_manager/*.cpp"
        "tests/cpp/program/*.cpp"
        "tests/cpp/rhi/common/*.cpp"
        "tests/cpp/struct/*.cpp"
        "tests/cpp/transforms/*.cpp"
        "tests/cpp/offline_cache/*.cpp")

if(QD_WITH_LLVM)
  file(GLOB QUADRANTS_TESTS_LLVM_SOURCE "tests/cpp/llvm/*.cpp")
  list(APPEND QUADRANTS_TESTS_SOURCE ${QUADRANTS_TESTS_LLVM_SOURCE})
endif()

if(QD_WITH_CUDA)
  file(GLOB QUADRANTS_TESTS_CUDA_SOURCE "tests/cpp/runtime/cuda/*.cpp")
  list(APPEND QUADRANTS_TESTS_SOURCE ${QUADRANTS_TESTS_CUDA_SOURCE})
endif()

add_executable(${TESTS_NAME} ${QUADRANTS_TESTS_SOURCE})
if (WIN32)
    # Output the executable to build/ instead of build/Debug/...
    set(TESTS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR})
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${TESTS_OUTPUT_DIR})
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${TESTS_OUTPUT_DIR})
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${TESTS_OUTPUT_DIR})
    set_target_properties(${TESTS_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${TESTS_OUTPUT_DIR})
    if (MSVC AND QD_GENERATE_PDB)
        target_compile_options(${TESTS_NAME} PRIVATE "/Zi")
        target_link_options(${TESTS_NAME} PRIVATE "/DEBUG")
        target_link_options(${TESTS_NAME} PRIVATE "/OPT:REF")
        target_link_options(${TESTS_NAME} PRIVATE "/OPT:ICF")
    endif()
endif()
target_link_libraries(${TESTS_NAME} PRIVATE quadrants_core)
target_link_libraries(${TESTS_NAME} PRIVATE gtest_main)
target_link_libraries(${TESTS_NAME} PRIVATE quadrants_common)

if (QD_WITH_BACKTRACE)
    target_link_libraries(${TESTS_NAME} PRIVATE ${BACKWARD_ENABLE})
endif()

if (QD_WITH_VULKAN)
  target_link_libraries(${TESTS_NAME} PRIVATE gfx_runtime)
endif()

if (QD_WITH_VULKAN)
  target_link_libraries(${TESTS_NAME} PRIVATE vulkan_rhi)
endif()

target_include_directories(${TESTS_NAME}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/volk
    ${PROJECT_SOURCE_DIR}/external/glad/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
  )

target_include_directories(${TESTS_NAME} SYSTEM
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
  )

if(LINUX)
    target_link_options(${TESTS_NAME} PUBLIC -Wl,--exclude-libs=ALL)
    target_link_options(${TESTS_NAME} PUBLIC -static-libgcc -static-libstdc++)
    target_link_libraries(${TESTS_NAME} PUBLIC stdc++fs)
endif()
add_test(NAME ${TESTS_NAME} COMMAND ${TESTS_NAME})
