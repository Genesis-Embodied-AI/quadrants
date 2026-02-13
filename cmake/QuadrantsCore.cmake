option(USE_STDCPP "Use -stdlib=libc++" OFF)
option(TI_WITH_LLVM "Build with LLVM backends" ON)              # wheel-tag: llvm
option(TI_WITH_METAL "Build with the Metal backend" ON)         # wheel-tag: mtl
option(TI_WITH_CUDA "Build with the CUDA backend" ON)           # wheel-tag: cu
option(TI_WITH_CUDA_TOOLKIT "Build with the CUDA toolkit" OFF)  # wheel-tag: cutk
option(TI_WITH_AMDGPU "Build with the AMDGPU backend" OFF)      # wheel-tag: amd
option(TI_WITH_VULKAN "Build with the Vulkan backend" OFF)      # wheel-tag: vk

# Force symbols to be 'hidden' by default so nothing is exported from the Quadrants
# library including the third-party dependencies.
# As Quadrants can be used by external projects, some of the internal dependencies
# such as Vulkan, etc. could be in conflict with the dependencies of those
# projects.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
# Suppress warnings from submodules introduced by the above symbol visibility change
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/python/quadrants/_lib)

if (TI_WITH_AMDGPU AND TI_WITH_CUDA)
    message(WARNING "Compiling CUDA and AMDGPU backends simultaneously")
endif()

if(UNIX AND NOT APPLE)
    # Handy helper for Linux
    # https://stackoverflow.com/a/32259072/12003165
    set(LINUX TRUE)
endif()

if (APPLE)
    if (TI_WITH_CUDA)
        set(TI_WITH_CUDA OFF)
        message(WARNING "CUDA backend not supported on OS X. Setting TI_WITH_CUDA to OFF.")
    endif()
    if (TI_WITH_AMDGPU)
        set(TI_WITH_AMDGPU OFF)
        message(WARNING "AMDGPU backend not supported on OS X. Setting TI_WITH_AMDGPU to OFF.")
    endif()
else()
    if (TI_WITH_METAL)
        set(TI_WITH_METAL OFF)
        message(WARNING "Metal backend only supported on OS X. Setting TI_WITH_METAL to OFF.")
    endif()
endif()

if (WIN32)
    if (TI_WITH_AMDGPU)
        set(TI_WITH_AMDGPU OFF)
        message(WARNING "AMDGPU backend not supported on Windows. Setting TI_WITH_AMDGPU to OFF.")
    endif()
endif()

if(TI_WITH_VULKAN)
    set(TI_WITH_GGUI ON)
endif()

if(NOT TI_WITH_LLVM)
    set(TI_WITH_CUDA OFF)
    set(TI_WITH_CUDA_TOOLKIT OFF)
endif()

file(GLOB QUADRANTS_CORE_SOURCE
    "quadrants/analysis/*.cpp" "quadrants/analysis/*.h"
    "quadrants/ir/*"
    "quadrants/jit/*"
    "quadrants/math/*"
    "quadrants/program/*"
    "quadrants/struct/*"
    "quadrants/system/*"
    "quadrants/transforms/*"
    "quadrants/platform/cuda/*" "quadrants/platform/amdgpu/*"
    "quadrants/platform/mac/*" "quadrants/platform/windows/*"
    "quadrants/codegen/*.cpp" "quadrants/codegen/*.h"
    "quadrants/runtime/*.h" "quadrants/runtime/*.cpp"
)

if(TI_WITH_LLVM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_LLVM")
endif()

if (TI_WITH_CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_CUDA")
  file(GLOB QUADRANTS_CUDA_RUNTIME_SOURCE "quadrants/runtime/cuda/runtime.cpp")
  list(APPEND QUADRANTS_CORE_SOURCE ${QUADRANTS_CUDA_RUNTIME_SOURCE})
endif()

if (TI_WITH_AMDGPU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_AMDGPU")
  file(GLOB QUADRANTS_AMDGPU_RUNTIME_SOURCE "quadrants/runtime/amdgpu/runtime.cpp")
  list(APPEND TAIHI_CORE_SOURCE ${QUADRANTS_AMDGPU_RUNTIME_SOURCE})
endif()

if (TI_WITH_METAL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_METAL")
endif()

if (TI_WITH_VULKAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_VULKAN")
endif ()

add_subdirectory(quadrants/rhi)

set(CORE_LIBRARY_NAME quadrants_core)
add_library(${CORE_LIBRARY_NAME} OBJECT ${QUADRANTS_CORE_SOURCE})

target_include_directories(${CORE_LIBRARY_NAME} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/include)
target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/SPIRV-Tools/include)
target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/PicoSHA2)
target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/eigen)
target_include_directories(${CORE_LIBRARY_NAME} PRIVATE external/FP16/include)

target_link_libraries(${CORE_LIBRARY_NAME} PUBLIC ti_device_api)

if(TI_WITH_LLVM)
    if(DEFINED ENV{LLVM_DIR})
        set(LLVM_DIR $ENV{LLVM_DIR})
        message("Getting LLVM_DIR=${LLVM_DIR} from the environment variable")
    endif()

    # http://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    if(${LLVM_PACKAGE_VERSION} VERSION_LESS "10.0")
        message(FATAL_ERROR "LLVM version < 10 is not supported")
    endif()
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    target_include_directories(${CORE_LIBRARY_NAME} PUBLIC ${LLVM_INCLUDE_DIRS})

    message("LLVM include dirs ${LLVM_INCLUDE_DIRS}")
    message("LLVM library dirs ${LLVM_LIBRARY_DIRS}")
    add_definitions(${LLVM_DEFINITIONS})

    llvm_map_components_to_libnames(llvm_libs
            Core
            ExecutionEngine
            InstCombine
            OrcJIT
            RuntimeDyld
            TransformUtils
            BitReader
            BitWriter
            Object
            ScalarOpts
            Support
            native
            Linker
            Target
            MC
            Passes
            ipo
            Analysis
            )

    if (APPLE AND "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
        llvm_map_components_to_libnames(llvm_aarch64_libs AArch64)
    endif()

    add_subdirectory(quadrants/codegen/cpu)
    add_subdirectory(quadrants/runtime/cpu)

    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE cpu_codegen)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE cpu_runtime)

    if (TI_WITH_CUDA)
        llvm_map_components_to_libnames(llvm_ptx_libs NVPTX)
        add_subdirectory(quadrants/codegen/cuda)
        add_subdirectory(quadrants/runtime/cuda)

        target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE cuda_codegen)
        target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE cuda_runtime)
    endif()

    if (TI_WITH_AMDGPU)
        llvm_map_components_to_libnames(llvm_amdgpu_libs AMDGPU)
        add_subdirectory(quadrants/codegen/amdgpu)
        add_subdirectory(quadrants/runtime/amdgpu)

        target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE amdgpu_codegen)
        target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE amdgpu_runtime)
    endif()

    add_subdirectory(quadrants/codegen/llvm)
    add_subdirectory(quadrants/runtime/llvm)
    add_subdirectory(quadrants/runtime/program_impls/llvm)

    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE llvm_program_impl)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE llvm_codegen)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE llvm_runtime)

    if (LINUX)
        # Remove symbols from llvm static libs
        foreach(LETTER ${llvm_libs})
            target_link_options(${CORE_LIBRARY_NAME} PUBLIC -Wl,--exclude-libs=lib${LETTER}.a)
        endforeach()
    endif()
endif()

if (TI_WITH_METAL OR TI_WITH_VULKAN)
    add_subdirectory(quadrants/runtime/program_impls/gfx)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE gfx_program_impl)
endif()

if (TI_WITH_METAL)
    add_subdirectory(quadrants/runtime/program_impls/metal)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE metal_program_impl)
endif()

if (TI_WITH_VULKAN)
    add_subdirectory(quadrants/runtime/program_impls/vulkan)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE vulkan_program_impl)
endif ()

add_subdirectory(quadrants/util)
add_subdirectory(quadrants/common)
add_subdirectory(quadrants/compilation_manager)

target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE compilation_manager)
target_link_libraries(${CORE_LIBRARY_NAME} PUBLIC quadrants_util)
target_link_libraries(${CORE_LIBRARY_NAME} PUBLIC quadrants_common)

if (TI_WITH_CUDA AND TI_WITH_CUDA_TOOLKIT)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDAToolkit ${CUDAToolkit_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_CUDA_TOOLKIT")
    target_include_directories(${CORE_LIBRARY_NAME} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${CORE_LIBRARY_NAME} PUBLIC CUDA::cupti)
endif()

# SPIR-V codegen is always there, regardless of Vulkan
set(SPIRV_SKIP_EXECUTABLES true)
set(SPIRV-Headers_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/SPIRV-Headers)
set(ENABLE_SPIRV_TOOLS_INSTALL OFF)
add_subdirectory(external/SPIRV-Tools)
add_subdirectory(quadrants/codegen/spirv)
add_subdirectory(quadrants/runtime/gfx)

if (TI_WITH_VULKAN OR TI_WITH_METAL)
  target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE spirv_codegen)
  target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE gfx_runtime)
endif()

if (TI_WITH_METAL)
  set(SPIRV_CROSS_CLI false)
  add_subdirectory(${PROJECT_SOURCE_DIR}/external/SPIRV-Cross ${PROJECT_BINARY_DIR}/external/SPIRV-Cross)
endif()


# Optional dependencies
if (APPLE)
    set(APPLE_FRAMEWORKS "")
    find_library(Foundation NAMES Foundation REQUIRED)
    find_library(Metal NAMES Metal REQUIRED)
    list(APPEND APPLE_FRAMEWORKS ${Foundation} ${Metal})
    if (NOT IOS)
        find_library(ApplicationServices NAMES ApplicationServices REQUIRED)
        find_library(Cocoa NAMES Cocoa REQUIRED)
        list(APPEND APPLE_FRAMEWORKS ${ApplicationServices} ${Cocoa})
    endif()
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE ${APPLE_FRAMEWORKS})
endif ()

if (LINUX)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE pthread)
    if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        # Avoid glibc dependencies
        if (TI_WITH_VULKAN)
            target_link_options(${CORE_LIBRARY_NAME} PRIVATE -Wl,--wrap=log2f)
        else()
            # Enforce compatibility with manylinux2014
            target_link_options(${CORE_LIBRARY_NAME} PRIVATE -Wl,--wrap=log2f -Wl,--wrap=exp2 -Wl,--wrap=log2 -Wl,--wrap=logf -Wl,--wrap=powf -Wl,--wrap=exp -Wl,--wrap=log -Wl,--wrap=pow)
        endif()
    endif()
elseif (WIN32)
    target_link_libraries(${CORE_LIBRARY_NAME} PRIVATE Winmm)
endif()



foreach (source IN LISTS QUADRANTS_CORE_SOURCE)
    file(RELATIVE_PATH source_rel ${CMAKE_CURRENT_LIST_DIR} ${source})
    get_filename_component(source_path "${source_rel}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach ()

if(TI_WITH_PYTHON)
    message("PYTHON_LIBRARIES: " ${PYTHON_LIBRARIES})
    set(CORE_WITH_PYBIND_LIBRARY_NAME quadrants_python)
    # NO_EXTRAS is required here to avoid llvm symbol error during build
    file(GLOB QUADRANTS_PYBIND_SOURCE
        "quadrants/python/*.cpp"
        "quadrants/python/*.h"
    )
    pybind11_add_module(${CORE_WITH_PYBIND_LIBRARY_NAME} NO_EXTRAS ${QUADRANTS_PYBIND_SOURCE})

    # Remove symbols from static libs: https://stackoverflow.com/a/14863432/12003165
    if (LINUX)
        target_link_options(${CORE_WITH_PYBIND_LIBRARY_NAME} PUBLIC -Wl,--exclude-libs=ALL)
        target_link_options(${CORE_WITH_PYBIND_LIBRARY_NAME} PUBLIC -static-libgcc -static-libstdc++)
        target_link_libraries(${CORE_WITH_PYBIND_LIBRARY_NAME} PUBLIC stdc++fs)
    endif()

    if (TI_WITH_BACKTRACE)
        # Defined by external/backward-cpp:
        # This will add libraries, definitions and include directories needed by backward
        # by setting each property on the target.
        target_link_libraries(${CORE_WITH_PYBIND_LIBRARY_NAME} PRIVATE ${BACKWARD_ENABLE})
    endif()

    target_link_libraries(${CORE_WITH_PYBIND_LIBRARY_NAME} PRIVATE ${CORE_LIBRARY_NAME})

    target_include_directories(${CORE_WITH_PYBIND_LIBRARY_NAME}
      PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/external/spdlog/include
        ${PROJECT_SOURCE_DIR}/external/eigen
        ${PROJECT_SOURCE_DIR}/external/volk
        ${PROJECT_SOURCE_DIR}/external/dlpack/include
        ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
        ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
        ${PROJECT_SOURCE_DIR}/external/FP16/include
      )
    target_include_directories(${CORE_WITH_PYBIND_LIBRARY_NAME} SYSTEM
      PRIVATE
        ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
      )

    # These commands should apply to the DLL that is loaded from python, not the OBJECT library.
    if (MSVC)
        set_property(TARGET ${CORE_WITH_PYBIND_LIBRARY_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
    endif ()

    if (WIN32)
        set_target_properties(${CORE_WITH_PYBIND_LIBRARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
    endif ()

    install(TARGETS ${CORE_WITH_PYBIND_LIBRARY_NAME}
            RUNTIME DESTINATION ${INSTALL_LIB_DIR}/core
            LIBRARY DESTINATION ${INSTALL_LIB_DIR}/core)
endif()

if (NOT APPLE)
    # For more background on what is slim_libdevice.10.bc, and why version 10, not 12.8
    # See https://github.com/Genesis-Embodied-AI/quadrants/issues/166#issuecomment-3289552564
    install(FILES ${CMAKE_SOURCE_DIR}/external/cuda_libdevice/slim_libdevice.10.bc
            DESTINATION ${INSTALL_LIB_DIR}/runtime)
endif()

if (TI_WITH_AMDGPU)
    # Install ROCm 7.0 libdevice files
    file(GLOB AMDGPU_BC_FILES_ROCM70 ${CMAKE_SOURCE_DIR}/external/amdgpu_libdevice_rocm70/*.bc)
    install(FILES ${AMDGPU_BC_FILES_ROCM70}
            DESTINATION ${INSTALL_LIB_DIR}/runtime_rocm70)
endif()
