# GLSL shader helpers

`workgroup.comp` implements the workgroup-index query in GLSL. `workgroup_spv.h` contains its compiled SPIR-V library, embedded in Quadrants so installed wheels do not require a shader compiler or an extra runtime file.

CMake compiles the shader automatically and writes `generated/workgroup_spv.h` under the SPIR-V build directory. Editing the shader rebuilds the header. Generated headers are not stored in Git.

Source builds require glslang with `--no-link` support (13.1 or newer; tested with 15.4.0). The build bootstrap uses pinned glslang 15.4.0 archives on Linux x86_64 and ARM64. It verifies their SHA-256 checksums before extraction and caches the compiler. These archives are built for the manylinux 2.28 x86_64 and manylinux 2.34 ARM64 containers. Windows and macOS use the Vulkan SDK compiler. An explicit `QD_GLSLANG_EXECUTABLE` setting takes precedence. For direct CMake builds, install the SDK and put its tools on PATH, set VULKAN_SDK, or pass `-DQD_GLSLANG_EXECUTABLE=/path/to/glslangValidator`. Installed wheels do not require glslang.

CMake invokes glslang directly with `--vn` to generate a C++ array in the header. The array contains the compiled SPIR-V words. The `--target-env vulkan1.0` option selects Vulkan 1.0 / SPIR-V 1.0 for compatibility with the oldest generated kernels.

Add another helper through the reusable CMake function:

```cmake
qd_add_shader_helper(spirv_codegen shaders/example.comp example_spv.h example_helper_spv)
```

The arguments name the consuming target, GLSL source, generated header, and C++ array. This function adds header generation, dependency tracking, and the generated include directory to the target. A GLSL source can export several helper functions; it does not require one file per function. Linking the new library and requesting its functions remain separate C++ integration steps.

`IRBuilder::get_work_group_id()` delegates to `call_glsl_u32()`. That reusable helper declares an imported `uint(uint)` function and passes its argument using GLSL's Function-storage pointer convention. `IRBuilder::finalize()` selects the workgroup library only when that import is used. `get_workgroup_shader_library()` prepares that library for the kernel. `link_spirv_modules()` links the supplied modules without selecting libraries. SPIRV-Tools resolves the function and includes the helper's `WorkgroupId` input in the kernel's entry-point interface. The existing optimizer inlines the call when optimization is enabled. Unoptimized calls remain valid.

The linker requires matching addressing models. The library uses only Input and Function pointers, so `get_workgroup_shader_library()` adjusts its module addressing model to match the kernel. The kernel already supplies any required physical-storage capability and extension. It does not change the helper's executable instructions.

C++ tests validate linking and optimization across SPIR-V versions and addressing models, including the Metal compiler configuration. C++ tests cover both unoptimized and optimized modules. Python tests execute repeated `impl.call_internal("workgroupId")` queries on Vulkan and Metal. This internal operation returns the x component as a signed 32-bit integer. It does not add a public Python API. Metal execution still requires Apple hardware.

The Linux compiler archives come from [release glslang-15.4.0-202610051822](https://github.com/Genesis-Embodied-AI/quadrants-sdk-builds/releases/tag/glslang-15.4.0-202610051822), built from glslang revision `8a85691a0740d390761a1008b4696f57facd02c4`. Normal Quadrants builds download these binaries; they do not build glslang from source. The archive names and checksums are pinned in `.github/workflows/scripts/qd_build/glslang.py`.
