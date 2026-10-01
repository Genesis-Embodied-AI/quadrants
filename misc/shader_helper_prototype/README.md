# Separately compiled shader helper prototype

This standalone experiment compiles `helper.comp` with glslang, generates a separate SPIR-V caller, and links the two. It does not modify Quadrants' compiler or depend on PR #949. The caller contains no implementation of the helper.

Requirements: Python 3.10+, `glslangValidator`, `spirv-as`, `spirv-dis`, `spirv-link`, `spirv-val`, and `spirv-opt`. Execution also needs the Python `vulkan` package and a discrete GPU with Vulkan 1.1 support. Run builds and checks on a cluster allocation, not the shared development box.

```bash
python misc/shader_helper_prototype/prototype.py --out /path/to/artifacts --run
```

Omit `--run` to compile, link, validate, and inspect without a GPU. Output files preserve every intermediate module, including the linked and optimized SPIR-V disassembly.

The helper uses GLSL's `gl_WorkGroupID`. The prototype adds an empty entry point for glslang, preserves uncalled functions, then removes that entry point and marks the helper as exported. The separately generated caller imports the function. This adapter is intentionally specific to this helper; it is not a general shader library system.

GLSL represents the scalar function argument as a pointer in Function storage. The caller matches that calling convention. The prototype adds the helper's `WorkgroupId` input to the linked kernel's entry-point interface if needed, validates the finished module for Vulkan, and optimizes it. It rejects any remaining function calls after optimization.

The Vulkan check dispatches a 7 × 3 × 2 grid with four invocations per workgroup. Every invocation writes the x, y, and z block indices plus a component selected by a varying dimension index. Python checks all 672 values against expectations. Both unoptimized and optimized kernels run, and the device must be a discrete GPU.

This is a feasibility experiment. It does not establish integration with Quadrants' IR builder, Metal support, arbitrary helper signatures, or general input/resource merging.
