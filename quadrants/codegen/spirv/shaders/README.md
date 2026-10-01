# GLSL workgroup helper

`workgroup.comp` implements the workgroup-index query in GLSL. `workgroup_spv.h` contains its compiled SPIR-V library, embedded in Quadrants so normal builds and installed wheels do not require a shader compiler or an extra runtime file.

Regenerate the header after changing the shader. Use glslang 15.4.0, which supports `--no-link`:

```bash
python quadrants/codegen/spirv/shaders/generate_workgroup.py
python quadrants/codegen/spirv/shaders/generate_workgroup.py --check
```

The script targets Vulkan 1.0 / SPIR-V 1.0 for compatibility with the oldest generated kernels. CMake verifies the GLSL source hash in the generated header. `--check` additionally recompiles the shader and compares the full generated header. A compiler-version change may alter the binary and should be reviewed along with its regenerated header.

`IRBuilder::get_work_group_id()` declares an imported function and passes its dimension argument using GLSL's Function-storage pointer convention. `IRBuilder::finalize()` links the library only when that import is used. SPIRV-Tools resolves the function and includes the helper's `WorkgroupId` input in the kernel's entry-point interface. The existing optimizer inlines the call when optimization is enabled. Unoptimized calls remain valid.

The linker requires matching addressing models. The library uses only Input and Function pointers, so the linker wrapper adjusts its module addressing model to match the kernel. The kernel already supplies any required physical-storage capability and extension. It does not change the helper's executable instructions.

C++ tests validate linking and optimization across SPIR-V versions and addressing models, including the Metal compiler configuration. Python tests execute `qd.block_idx()` with optimization enabled and disabled on Vulkan and Metal. Metal execution still requires Apple hardware.
