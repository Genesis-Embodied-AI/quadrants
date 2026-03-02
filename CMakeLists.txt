#*********************************************************************
# The Quadrants Programming Language
#*********************************************************************

cmake_minimum_required(VERSION 3.17)

project(quadrants)

include("cmake/utils.cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILECOMMANDS ON)

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND git rev-parse --short HEAD
  RESULT_VARIABLE SHORT_HASH_RESULT
  OUTPUT_VARIABLE QD_COMMIT_SHORT_HASH)
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND git rev-parse HEAD
  RESULT_VARIABLE SHORT_HASH_RESULT
  OUTPUT_VARIABLE QD_COMMIT_HASH)
string(STRIP ${QD_COMMIT_HASH} QD_COMMIT_HASH)
string(STRIP ${QD_COMMIT_SHORT_HASH} QD_COMMIT_SHORT_HASH)

message("Quadrants Version ${QD_VERSION_MAJOR}.${QD_VERSION_MINOR}.${QD_VERSION_PATCH}")
message("       Commit ${QD_COMMIT_HASH}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
            FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(QUADRANTS_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/")

if (WIN32)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules")
else ()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_ROOT}/Modules")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/")
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# This compiles all the libraries with -fPIC, which is critical to link a static
# library into a shared lib.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


option(USE_LLD "Use lld (from llvm) linker" OFF)
option(USE_MOLD "Use mold (A Modern Linker)" OFF)
option(QD_WITH_BACKTRACE "Use backward-cpp to print out C++ stack trace upon failure" OFF)  # wheel-tag: bt
option(QD_GENERATE_PDB "Generate Program Database (PDB) files (will make compilation uncacheable)" OFF)
option(QD_WITH_LTO "Enable Link Time Optimization (LTO) (affects Windows + MSVC for now)" OFF)  # wheel-tag: lto

if((LINUX OR APPLE) AND NOT IOS)
      # (penguinliong) Not compatible with -fembed-bitcode. Not sure where it
      # comes from; probably a XCode preset?
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
endif()

if(APPLE)
    add_definitions(-DOSX_DEPLOYMENT_TARGET=11.0)
    add_definitions(-DTARGET_OS_OSX)
endif()

if (USE_LLD)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=lld")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")

    if (WIN32)
        if (CMAKE_BUILD_TYPE EQUAL "RelWithDebInfo")
            # -debug for LLD generates PDB files
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-debug")
            set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-debug")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-debug")
        endif()
    endif()
endif()

if (USE_MOLD)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=mold")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=mold")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold")
endif()

if (WIN32)
  # For `Debug` configs MSVC links to a debuggable runtime by default which has
  # symbol conflicts with the prebuilt LLVM in `Release`. We should be providing
  # prebuilt LLVMs for both `Debug` and `Release` but LLVM 10 cannot be built by
  # MSVC in `Debug` config because MSVC would try to fill uninitialize memory
  # with `0xCC` but it too breaks `LLVMTableGen` which is depended on by almost
  # every component in LLVM.
  message("CMAKE_MSVC_RUNTIME_LIBRARY: ${CMAKE_MSVC_RUNTIME_LIBRARY}")
endif()

# No support of Python for any case quadrants is integrated
# in another project as submodule.
set(PYBIND11_VERSION "2.13.6" CACHE STRING "Pybind11 version to use")
option(QD_WITH_PYTHON "Build with Python language binding" ON)
if (QD_WITH_PYTHON)
    include(cmake/PythonNumpyPybind11.cmake)
endif()

if (QD_WITH_BACKTRACE)
  add_subdirectory(external/backward_cpp)
endif()

if (QD_DISTRIBUTED_COMPILE)
    include(cmake/Distributed.cmake)
endif()

include(cmake/QuadrantsCXXFlags.cmake)
include(cmake/QuadrantsCore.cmake)

option(QD_BUILD_TESTS "Build the CPP tests" OFF)

if (QD_BUILD_TESTS)
  add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
  include(cmake/QuadrantsTests.cmake)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" _BUILD_TYPE)
message("C++ Flags: ${CMAKE_CXX_FLAGS_${_BUILD_TYPE}} ${CMAKE_CXX_FLAGS}")
message("Build type: ${CMAKE_BUILD_TYPE}")

if (NOT QD_WITH_CUDA)
    set(CUDA_TOOLKIT_ROOT_DIR "")
endif()

if (QD_WITH_CUDA)
    set(CUDA_ARCH "cuda")
endif()

if (QD_WITH_AMDGPU)
    set(AMDGPU_ARCH "amdgpu")
endif()

if (QD_WITH_LLVM)
    # Setup CLANG_EXECUTABLE
    if (CLANG_EXECUTABLE)
      message("CLANG_EXECUTABLE defined: ${CLANG_EXECUTABLE}")
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
      set (CLANG_EXECUTABLE ${CMAKE_CXX_COMPILER})
      message("Clang executable using host compiler ${CLANG_EXECUTABLE}")
    else()
      find_program(CLANG_EXECUTABLE NAMES clang-22 clang-20 clang)
      message("Clang executable found at ${CLANG_EXECUTABLE}")
    endif()

    if (NOT CLANG_EXECUTABLE)
      message(FATAL_ERROR "Cannot find any clang executable.")
    endif()

    macro(check_clang_version)
      execute_process(COMMAND ${CLANG_EXECUTABLE} --version OUTPUT_VARIABLE CLANG_VERSION_OUTPUT)
      string(REGEX MATCH "([0-9]+)\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION "${CLANG_VERSION_OUTPUT}")

      message("${CLANG_EXECUTABLE} --version: ${CLANG_VERSION}")

      set(CLANG_VERSION_MAJOR "${CMAKE_MATCH_1}")
    endmacro()

    if (APPLE)
      if(CMAKE_OSX_SYSROOT)
        set(CLANG_OSX_FLAGS "-isysroot" "${CMAKE_OSX_SYSROOT}")
      else()
        set(CLANG_OSX_FLAGS "")
      endif()
    endif()

    # Highest clang version that we've tested
    set(CLANG_HIGHEST_VERSION "15")

    check_clang_version()

    add_subdirectory(quadrants/runtime/llvm/runtime_module)
endif()

configure_file(quadrants/common/version.h.in ${CMAKE_SOURCE_DIR}/quadrants/common/version.h)
configure_file(quadrants/common/commit_hash.h.in ${CMAKE_SOURCE_DIR}/quadrants/common/commit_hash.h)

option(QD_WITH_GRAPHVIZ "generate dependency graphs between targets" OFF)  # wheel-tag: viz
if (QD_WITH_GRAPHVIZ)
  set(GRAPHVIZ_GRAPH_NAME "ti_targets")
  add_custom_target(graphviz ALL
      COMMAND cp ${CMAKE_SOURCE_DIR}/cmake/CMakeGraphVizOptions.cmake .
      COMMAND ${CMAKE_COMMAND} "--graphviz=${GRAPHVIZ_GRAPH_NAME}.dot" .
      COMMAND dot -Tpng ${GRAPHVIZ_GRAPH_NAME}.dot -o ${GRAPHVIZ_GRAPH_NAME}.png
      COMMAND cp ti_targets.png ${CMAKE_SOURCE_DIR}/build
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
endif()
