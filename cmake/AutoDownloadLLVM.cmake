# Auto-provision LLVM for the scikit-build-core (pip / uv) build.
#
# The classic build.py flow downloads a prebuilt LLVM and exports LLVM_DIR *before* invoking CMake.
# A plain `pip install -e .` / `uv pip install -e .` does not run build.py, so without this hook
# find_package(LLVM) silently falls back to whatever system LLVM it can find (usually the wrong
# major version) and the LLVM-22-era sources fail to compile. This reproduces build.py's LLVM
# provisioning during CMake configure by invoking the same download_llvm.py, so the editable / wheel
# build works with no separate manual step.
#
# Active only for the scikit-build-core build (SKBUILD_PROJECT_NAME defined), where the build
# backend's isolated environment is guaranteed to provide download_llvm.py's dependencies
# (requests, tqdm) via [build-system].requires in pyproject.toml. It is skipped whenever the user
# already points the build at an LLVM via LLVM_DIR or LLVM_ROOT (env var or cache). Disable entirely
# with -DQD_AUTO_DOWNLOAD_LLVM=OFF.

option(QD_AUTO_DOWNLOAD_LLVM
       "Auto-download a prebuilt LLVM when LLVM_DIR/LLVM_ROOT are unset (scikit-build-core build only)"
       ON)

function(qd_auto_download_llvm)
    if(NOT QD_AUTO_DOWNLOAD_LLVM)
        return()
    endif()
    # Leave the classic build.py / direct-CMake flows untouched: they set LLVM_DIR themselves and do
    # not guarantee the downloader's Python dependencies are importable.
    if(NOT DEFINED SKBUILD_PROJECT_NAME)
        return()
    endif()
    # Respect any user-supplied LLVM location.
    if(DEFINED ENV{LLVM_DIR} OR DEFINED LLVM_DIR OR DEFINED ENV{LLVM_ROOT} OR DEFINED LLVM_ROOT)
        return()
    endif()

    if(Python_EXECUTABLE)
        set(_py "${Python_EXECUTABLE}")
    elseif(PYTHON_EXECUTABLE)
        set(_py "${PYTHON_EXECUTABLE}")
    else()
        set(_py "python")
    endif()

    message(STATUS "LLVM_DIR/LLVM_ROOT unset; auto-downloading a prebuilt LLVM via download_llvm.py ...")
    execute_process(
        COMMAND "${_py}" "${CMAKE_SOURCE_DIR}/download_llvm.py"
        OUTPUT_VARIABLE _out
        RESULT_VARIABLE _rc
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR
            "Automatic LLVM download failed (rc=${_rc}). Point the build at an LLVM with "
            "-DLLVM_DIR=... (or LLVM_ROOT), run `python download_llvm.py` manually, or disable "
            "auto-download with -DQD_AUTO_DOWNLOAD_LLVM=OFF.")
    endif()

    # download_llvm.py prints the install root as its final stdout line (banners/progress go to stderr).
    string(REGEX REPLACE "\r?\n" ";" _lines "${_out}")
    list(FILTER _lines EXCLUDE REGEX "^[ \t]*$")
    if(NOT _lines)
        message(FATAL_ERROR "download_llvm.py produced no LLVM path on stdout.")
    endif()
    list(GET _lines -1 _root)
    string(STRIP "${_root}" _root)

    # The downloader unzips with a pure-Python extractor that drops the executable bit, so the bundled
    # clang/llvm tools come out non-executable (the CI's prerequisites script likewise `chmod +x`es
    # them). Restore exec permission on bin/* so the device-runtime clang invocation below can run.
    if(NOT WIN32)
        file(GLOB _bins "${_root}/bin/*")
        if(_bins)
            if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
                file(CHMOD ${_bins} PERMISSIONS
                     OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
            else()
                execute_process(COMMAND chmod +x ${_bins})
            endif()
        endif()
    endif()

    # find_package(LLVM CONFIG) wants the directory that holds LLVMConfig.cmake, not the install root.
    set(_cfg "${_root}/lib/cmake/llvm")
    if(NOT EXISTS "${_cfg}/LLVMConfig.cmake")
        file(GLOB_RECURSE _found "${_root}/LLVMConfig.cmake")
        if(_found)
            list(GET _found 0 _first)
            get_filename_component(_cfg "${_first}" DIRECTORY)
        endif()
    endif()
    set(ENV{LLVM_DIR} "${_cfg}")
    message(STATUS "Auto-downloaded LLVM: LLVM_DIR=$ENV{LLVM_DIR}")

    # Compile the device-runtime bitcode with the downloaded LLVM's own clang. The host links LLVM 22,
    # and emitting runtime .bc with a mismatched system clang risks bitcode incompatibility. Honoured
    # by the `if (CLANG_EXECUTABLE)` branch in the top-level CMakeLists, ahead of its find_program
    # fallback (which would otherwise prefer an older system clang-NN that happens to be installed).
    if(NOT CLANG_EXECUTABLE AND NOT DEFINED ENV{CLANG_EXECUTABLE})
        foreach(_c clang clang.exe)
            if(EXISTS "${_root}/bin/${_c}")
                set(CLANG_EXECUTABLE "${_root}/bin/${_c}"
                    CACHE FILEPATH "Clang from the auto-downloaded LLVM" FORCE)
                message(STATUS "Using clang from auto-downloaded LLVM: ${CLANG_EXECUTABLE}")
                break()
            endif()
        endforeach()
    endif()
endfunction()

qd_auto_download_llvm()
