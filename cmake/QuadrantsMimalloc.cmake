# Route C++ operator new / delete in the Quadrants binaries (quadrants_python and the C++ tests) to mimalloc.
#
# Windows only: the NT heap is much slower than glibc malloc for the many small, short-lived allocations made by the
# IR passes, and injecting mimalloc into the whole process made Windows kernel compilation 15-17% faster, while
# LD_PRELOAD-ing it on Linux made no measurable difference.
#
# The replacement operators are linked into the final DLL / EXE, so they only affect allocations made by code linked
# into that binary (Quadrants, LLVM, nanobind, spdlog, ...). Plain malloc / free, and other DLLs, still use the CRT heap.
#
# mimalloc is compiled from its single-file amalgamation rather than via add_subdirectory(), because mimalloc's install
# rules are unconditional and would ship its headers and libraries inside the wheel.

option(QD_USE_MIMALLOC "Use mimalloc for C++ operator new / delete (Windows only)" ON)

if (NOT (WIN32 AND QD_USE_MIMALLOC))
    return()
endif()

set(QD_MIMALLOC_DIR ${PROJECT_SOURCE_DIR}/external/mimalloc)
if (NOT EXISTS ${QD_MIMALLOC_DIR}/src/static.c)
    message(FATAL_ERROR "external/mimalloc is missing; run `git submodule update --init external/mimalloc`, "
                        "or configure with -DQD_USE_MIMALLOC=OFF")
endif()

add_library(qd_mimalloc STATIC ${QD_MIMALLOC_DIR}/src/static.c)
# mimalloc's own build compiles as C++ under MSVC / clang-cl, to use C++ atomics.
set_source_files_properties(${QD_MIMALLOC_DIR}/src/static.c PROPERTIES LANGUAGE CXX)
target_include_directories(qd_mimalloc PUBLIC ${QD_MIMALLOC_DIR}/include)
target_compile_definitions(qd_mimalloc PRIVATE MI_STATIC_LIB)
target_link_libraries(qd_mimalloc PRIVATE psapi shell32 user32 advapi32 bcrypt)
if (MSVC)
    target_compile_options(qd_mimalloc PRIVATE /Zc:__cplusplus /w)
endif()

target_sources(${CORE_LIBRARY_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/quadrants/system/mimalloc/new_delete.cpp)
target_link_libraries(${CORE_LIBRARY_NAME} PUBLIC qd_mimalloc)
target_compile_definitions(${CORE_LIBRARY_NAME} PUBLIC QD_USE_MIMALLOC)
