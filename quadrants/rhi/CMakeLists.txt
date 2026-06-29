# ./quadrants/rhi/CMakeLists.txt

set(QUADRANTS_DEVICE_API qd_device_api)
add_library(${QUADRANTS_DEVICE_API} STATIC)
target_sources(${QUADRANTS_DEVICE_API}
  PRIVATE
    arch.cpp
    device.cpp
    device_capability.cpp
  )

target_include_directories(${QUADRANTS_DEVICE_API}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
  )

target_include_directories(${QUADRANTS_DEVICE_API} SYSTEM
    PRIVATE
      ${PROJECT_SOURCE_DIR}/external/spdlog/include
    )

if (QD_WITH_METAL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DQD_WITH_METAL")
    add_subdirectory(metal)
    target_link_libraries(${QUADRANTS_DEVICE_API} PRIVATE metal_rhi)
endif()

if (QD_WITH_VULKAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DQD_WITH_VULKAN")
    if (APPLE)
        # MoltenVK comes from the LunarG Vulkan SDK. Sources are tried in order: explicit `$MOLTENVK_DIR`
        # (set by `build.py --shell`), then `$VULKAN_SDK/lib` (LunarG's globally-installed layout, also what
        # `build.py --shell` exports), then the previously-extracted SDK under the `qd-build-cache` that
        # `build.py --shell` populated on a prior invocation - this last fallback is what lets a plain
        # `cmake .` reconfigure without re-entering the build shell every time. No legacy-pin fallback on
        # purpose: the prior third-party dylib predates `VK_KHR_buffer_device_address` and is unusable for
        # the on-device adstack sizer.
        set(_MVK_CANDIDATES "")
        if(DEFINED ENV{MOLTENVK_DIR})
            list(APPEND _MVK_CANDIDATES "$ENV{MOLTENVK_DIR}")
        endif()
        if(DEFINED ENV{VULKAN_SDK})
            list(APPEND _MVK_CANDIDATES "$ENV{VULKAN_SDK}/lib")
        endif()
        # `build.py --shell` extracts the SDK to `~/.cache/qd-build-cache/vulkan-macos-<VER>/macOS/lib`.
        # Glob for any version under that directory so developers don't have to re-enter the build shell.
        file(GLOB _MVK_CACHE_CANDIDATES "$ENV{HOME}/.cache/qd-build-cache/vulkan-macos-*/macOS/lib")
        list(APPEND _MVK_CANDIDATES ${_MVK_CACHE_CANDIDATES})
        set(MOLTEN_VK "")
        foreach(_dir ${_MVK_CANDIDATES})
            if(EXISTS "${_dir}/libMoltenVK.dylib")
                set(MOLTEN_VK "${_dir}/libMoltenVK.dylib")
                break()
            endif()
        endforeach()
        if(NOT MOLTEN_VK)
            message(FATAL_ERROR
                "MoltenVK: cannot locate `libMoltenVK.dylib`. Searched: `$MOLTENVK_DIR`, `$VULKAN_SDK/lib`, "
                "and `~/.cache/qd-build-cache/vulkan-macos-*/macOS/lib` (populated by `build.py --shell`). "
                "Run `./build.py --shell` once to fetch LunarG's Vulkan SDK, or install the LunarG Vulkan "
                "SDK globally, before configuring Quadrants with `QD_WITH_VULKAN=ON`. The adstack sizer "
                "shader requires a MoltenVK with `VK_KHR_buffer_device_address` (>= 1.2.1); shipping an "
                "older pin would hard-error every reverse-mode kernel at launch time.")
        endif()
        message(STATUS "MoltenVK: using LunarG Vulkan SDK copy at ${MOLTEN_VK}")
        configure_file(${MOLTEN_VK} ${CMAKE_BINARY_DIR}/libMoltenVK.dylib COPYONLY)
        install(FILES ${CMAKE_BINARY_DIR}/libMoltenVK.dylib DESTINATION ${INSTALL_LIB_DIR}/runtime)
    endif()
    add_subdirectory(vulkan)

    # TODO: this dependency is here because program.cpp includes vulkan_program.h
    # Should be removed
    target_link_libraries(${QUADRANTS_DEVICE_API} PUBLIC vulkan_rhi)
endif ()

if(QD_WITH_LLVM)
    add_subdirectory(cpu)
    target_link_libraries(${QUADRANTS_DEVICE_API} PUBLIC cpu_rhi)

    if (QD_WITH_CUDA)
        add_subdirectory(cuda)
        target_link_libraries(${QUADRANTS_DEVICE_API} PUBLIC cuda_rhi)
    endif()

    if (QD_WITH_AMDGPU)
        add_subdirectory(amdgpu)
        target_link_libraries(${QUADRANTS_DEVICE_API} PUBLIC amdgpu_rhi)
    endif()

    add_subdirectory(llvm)
    target_link_libraries(${QUADRANTS_DEVICE_API} PUBLIC llvm_rhi)
endif()

add_subdirectory(interop)
target_link_libraries(${QUADRANTS_DEVICE_API} PUBLIC interop_rhi)

add_subdirectory(common)
target_link_libraries(${QUADRANTS_DEVICE_API} PUBLIC common_rhi)

# Generate shared library
add_library(qd_device_api_shared SHARED public_device.h)
target_link_libraries(qd_device_api_shared PUBLIC ${QUADRANTS_DEVICE_API})

# When building targets on Windows using Ninja + MSVC, the linker requires at least 1 object file
# to work properly, else link.exe would complain about LNK4001 warning, and fail afterwards.
# Adding a dummy file to workaround this.
target_sources(qd_device_api_shared PRIVATE dummy.cpp)
