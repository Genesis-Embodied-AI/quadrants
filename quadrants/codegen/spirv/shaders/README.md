# GLSL shader helpers

`workgroup.comp` implements the workgroup-index query in GLSL. `workgroup_spv.h` contains its compiled SPIR-V library, embedded in Quadrants so installed wheels do not require a shader compiler or an extra runtime file.

CMake compiles the shader automatically and writes `generated/workgroup_spv.h` under the SPIR-V build directory. Editing the shader or its generator rebuilds the header. Generated headers are not stored in Git.

Source builds require glslang with `--no-link` support (13.1 or newer; tested with 15.4.0). The build bootstrap installs the Vulkan SDK on Linux, macOS, and Windows. For direct CMake builds, install the SDK and put its tools on PATH, set VULKAN_SDK, or pass `-DQD_GLSLANG_EXECUTABLE=/path/to/glslangValidator`. Installed wheels do not require glslang.

`compile_shader.py` accepts `--input`, `--output`, and `--symbol` arguments. The symbol names the generated C++ array in the `quadrants::lang::spirv` namespace. The script does not hardcode a shader file or array name. Its optional `--compiler` argument selects glslang. Its optional `--target-env` argument defaults to Vulkan 1.0 / SPIR-V 1.0 for compatibility with the oldest generated kernels.

Add another helper through the reusable CMake function:

```cmake
qd_add_shader_helper(spirv_codegen shaders/example.comp example_spv.h example_helper_spv)
```

The arguments name the consuming target, GLSL source, generated header, and C++ array. This function adds header generation, dependency tracking, and the generated include directory to the target. A GLSL source can export several helper functions; it does not require one file per function. Linking the new library and requesting its functions remain separate C++ integration steps.

`IRBuilder::get_work_group_id()` delegates to `call_glsl_u32_helper()`. That reusable helper declares an imported `uint(uint)` function and passes its argument using GLSL's Function-storage pointer convention. `IRBuilder::finalize()` calls `link_shader_helpers()` to link the library only when that import is used. SPIRV-Tools resolves the function and includes the helper's `WorkgroupId` input in the kernel's entry-point interface. The existing optimizer inlines the call when optimization is enabled. Unoptimized calls remain valid.

The linker requires matching addressing models. The library uses only Input and Function pointers, so the linker wrapper adjusts its module addressing model to match the kernel. The kernel already supplies any required physical-storage capability and extension. It does not change the helper's executable instructions.

C++ tests validate linking and optimization across SPIR-V versions and addressing models, including the Metal compiler configuration. C++ tests cover both unoptimized and optimized modules. Python tests execute repeated `impl.call_internal("workgroupId")` queries on Vulkan and Metal. This internal operation returns the x component as a signed 32-bit integer. It does not add a public Python API. Metal execution still requires Apple hardware.
