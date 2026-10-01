# Separately compiled shader helper prototype

This standalone experiment compiles `helper.comp` with glslang, generates a separate SPIR-V caller, and links the two. It does not modify Quadrants' compiler or depend on PR #949. The caller contains no implementation of the helper.

Requirements: Python 3.10+, `glslangValidator` with `--no-link` support (tested with 15.4.0), `spirv-as`, `spirv-dis`, `spirv-link`, `spirv-val`, and `spirv-opt`. Execution also needs the Python `vulkan==1.3.275.1` package and a discrete GPU with Vulkan 1.1 support. Run builds and checks on a cluster allocation, not the shared development box.

```bash
python misc/shader_helper_prototype/prototype.py --out /path/to/artifacts --run
```

Omit `--run` to compile, link, validate, and inspect without a GPU. Output files preserve every intermediate module, including the linked and optimized SPIR-V disassembly.

The helper uses GLSL's `gl_WorkGroupID`. glslang's `--no-link` option exports the function as a SPIR-V library without a dummy entry point or assembly edits. The separately generated caller imports the function by name.

GLSL represents the scalar function argument as a pointer in Function storage. The caller matches that calling convention. The prototype checks that the linker includes the helper's `WorkgroupId` input in the kernel's entry-point interface, validates the finished module for Vulkan, and optimizes it. It rejects any remaining function calls after optimization.

The Vulkan check dispatches a 7 × 3 × 2 grid with four invocations per workgroup. Every invocation writes the x, y, and z block indices plus a component selected by a varying dimension index. Python checks all 672 values against expectations. Both unoptimized and optimized kernels run, and the device must be a discrete GPU.

This is a feasibility experiment. It does not establish integration with Quadrants' IR builder, Metal support, arbitrary helper signatures, or general input/resource merging.

## Observed results

Validated on 2026-10-01 using glslang 15.4.0, SPIRV-Tools v2025.3, and the Python Vulkan bindings 1.3.275.1. Compilation ran on `cpu-mid`; execution ran on an `rtx-mid` NVIDIA RTX PRO 6000 Blackwell Server Edition.

- Separate library and caller modules passed SPIR-V validation.
- Linking included the library's `WorkgroupId` input automatically. No manual interface repair was needed.
- Both linked and optimized kernels passed Vulkan 1.1 SPIR-V validation.
- Optimization removed all four helper calls. Constant dimensions became direct component reads at indices 0, 1, and 2; the varying dimension remained a dynamic component read.
- Both kernels returned all 672 expected values across 168 invocations in 42 workgroups.
- Repository pre-commit hooks passed for the prototype files.

This demonstrates that the readable GLSL helper can be compiled separately and linked into generated SPIR-V without a handwritten implementation of its body. The generated caller is standalone assembly, not output from Quadrants' existing IR builder. Production integration still needs build/package support, call generation with matching parameter types, and validation of supported SPIR-V tool versions and Metal translation.
