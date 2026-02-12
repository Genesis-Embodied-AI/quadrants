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

if (TI_WITH_METAL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_METAL")
    add_subdirectory(metal)
    target_link_libraries(${QUADRANTS_DEVICE_API} PRIVATE metal_rhi)
endif()

if (TI_WITH_VULKAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_WITH_VULKAN")
    if (APPLE)
        # The latest Molten-vk v1.2.0 and v1.1.11 breaks GGUI: mpm3d_ggui.py
        # So we have to manually download and install Molten-vk v1.10.0
        #
        # Uncomment the following lines if the mpm3d_ggui.py runs well with the latest Molten-vk
        #find_library(MOLTEN_VK libMoltenVK.dylib PATHS $HOMEBREW_CELLAR/molten-vk $VULKAN_SDK REQUIRED)
        #configure_file(${MOLTEN_VK} ${CMAKE_BINARY_DIR}/libMoltenVK.dylib COPYONLY)
        #message(STATUS "MoltenVK library ${MOLTEN_VK}")

        if(NOT EXISTS ${CMAKE_BINARY_DIR}/libMoltenVK.dylib)
            execute_process(COMMAND curl -L -o ${CMAKE_BINARY_DIR}/libMoltenVK.zip https://github.com/taichi-dev/taichi_assets/files/9977436/libMoltenVK.dylib.zip)
            execute_process(COMMAND tar -xf ${CMAKE_BINARY_DIR}/libMoltenVK.zip --directory ${CMAKE_BINARY_DIR})
        endif()
        install(FILES ${CMAKE_BINARY_DIR}/libMoltenVK.dylib DESTINATION ${INSTALL_LIB_DIR}/runtime)
    endif()
    add_subdirectory(vulkan)

    # TODO: this dependency is here because program.cpp includes vulkan_program.h
    # Should be removed
    target_link_libraries(${QUADRANTS_DEVICE_API} PUBLIC vulkan_rhi)
endif ()

if(TI_WITH_LLVM)
    add_subdirectory(cpu)
    target_link_libraries(${QUADRANTS_DEVICE_API} PUBLIC cpu_rhi)

    if (TI_WITH_CUDA)
        add_subdirectory(cuda)
        target_link_libraries(${QUADRANTS_DEVICE_API} PUBLIC cuda_rhi)
    endif()

    if (TI_WITH_AMDGPU)
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
