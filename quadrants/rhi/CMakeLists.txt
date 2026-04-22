# ./quadrants/rhi/CMakeLists.txt

set(QUADRANTS_DEVICE_API ti_device_api)
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
        # MoltenVK comes from the LunarG Vulkan SDK (`./build.py --shell` handles the download and exports
        # `MOLTENVK_DIR` pointing at `$VULKAN_SDK/lib`, where LunarG drops `libMoltenVK.dylib` flat). There
        # is no legacy-pin fallback on purpose: the prior third-party dylib predates
        # `VK_KHR_buffer_device_address` and is unusable for the on-device adstack sizer. A build without a
        # real SDK hard-fails here rather than shipping a Vulkan runtime that hard-errors every
        # reverse-mode kernel at launch time.
        if(DEFINED ENV{MOLTENVK_DIR})
            set(_MVK_ROOT "$ENV{MOLTENVK_DIR}")
        elseif(DEFINED ENV{VULKAN_SDK})
            set(_MVK_ROOT "$ENV{VULKAN_SDK}/lib")
        else()
            set(_MVK_ROOT "")
        endif()
        set(MOLTEN_VK "")
        if(_MVK_ROOT)
            find_file(MOLTEN_VK libMoltenVK.dylib NO_DEFAULT_PATH PATHS "${_MVK_ROOT}")
        endif()
        if(NOT MOLTEN_VK)
            message(FATAL_ERROR
                "MoltenVK: neither `$MOLTENVK_DIR` nor `$VULKAN_SDK` is set, or no `libMoltenVK.dylib` is "
                "present under them. On macOS, run `./build.py --shell` (fetches LunarG's Vulkan SDK and "
                "exports both vars), or install the LunarG Vulkan SDK globally, before configuring "
                "Quadrants with `QD_WITH_VULKAN=ON`. The adstack sizer shader requires a MoltenVK with "
                "`VK_KHR_buffer_device_address` (>= 1.2.1); shipping an older pin would hard-error every "
                "reverse-mode kernel at launch time.")
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
add_library(ti_device_api_shared SHARED public_device.h)
target_link_libraries(ti_device_api_shared PUBLIC ${QUADRANTS_DEVICE_API})

# When building targets on Windows using Ninja + MSVC, the linker requires at least 1 object file
# to work properly, else link.exe would complain about LNK4001 warning, and fail afterwards.
# Adding a dummy file to workaround this.
target_sources(ti_device_api_shared PRIVATE dummy.cpp)
