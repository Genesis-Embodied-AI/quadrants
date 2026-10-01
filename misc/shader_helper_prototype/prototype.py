"""Compile a GLSL helper separately, link a generated caller, and inspect the result.

This deliberately exercises module integration without modifying Quadrants' compiler.
The small text adapter is specific to this helper, not a general SPIR-V importer.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.check_output(list(map(str, args)), text=True)


def export_helper(assembly):
    """Remove glslang's required dummy entry point and export the retained helper."""
    helper = re.search(r'OpName (%\w+) "get_work_group_id\(', assembly).group(1)
    main = re.search(r'OpEntryPoint GLCompute (%\w+)', assembly).group(1)
    lines = []
    skipping_main = False
    for line in assembly.splitlines():
        if re.search(rf"{re.escape(main)} = OpFunction\b", line):
            skipping_main = True
        if skipping_main:
            if "OpFunctionEnd" in line:
                skipping_main = False
            continue
        if "OpEntryPoint" in line or "OpExecutionMode" in line or f"OpName {main} " in line:
            continue
        lines.append(line)
        if "OpCapability Shader" in line:
            lines.append("OpCapability Linkage")
        if "OpDecorate" in line and "BuiltIn WorkgroupId" in line:
            lines.append(f'OpDecorate {helper} LinkageAttributes "get_work_group_id" Export')
    return "\n".join(lines) + "\n"


def caller_assembly():
    """Generate a caller that imports the helper but never reads WorkgroupId itself.

    Dispatch is (7, 3, 2), local size is (4, 1, 1). Each invocation writes four
    values: helper(0), helper(1), helper(2), and helper(global_x % 3).
    GLSL uses a Function-storage pointer for this scalar parameter.
    """
    assembly = '''OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid
OpExecutionMode %main LocalSize 4 1 1
OpDecorate %get_id LinkageAttributes "get_work_group_id" Import
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %array ArrayStride 4
OpMemberDecorate %buffer 0 Offset 0
OpDecorate %buffer Block
OpDecorate %output DescriptorSet 0
OpDecorate %output Binding 0
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%input_v3 = OpTypePointer Input %v3uint
%gid = OpVariable %input_v3 Input
%array = OpTypeRuntimeArray %uint
%buffer = OpTypeStruct %array
%buffer_ptr = OpTypePointer StorageBuffer %buffer
%output = OpVariable %buffer_ptr StorageBuffer
%output_uint_ptr = OpTypePointer StorageBuffer %uint
%param_ptr = OpTypePointer Function %uint
%main_type = OpTypeFunction %void
%helper_type = OpTypeFunction %uint %param_ptr
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%two = OpConstant %uint 2
%three = OpConstant %uint 3
%four = OpConstant %uint 4
%width = OpConstant %uint 28
%get_id = OpFunction %uint None %helper_type
%import_arg = OpFunctionParameter %param_ptr
OpFunctionEnd
%main = OpFunction %void None %main_type
%entry = OpLabel
%arg = OpVariable %param_ptr Function
%global = OpLoad %v3uint %gid
%x = OpCompositeExtract %uint %global 0
%y = OpCompositeExtract %uint %global 1
%z = OpCompositeExtract %uint %global 2
%z_offset = OpIMul %uint %z %three
%row = OpIAdd %uint %y %z_offset
%row_offset = OpIMul %uint %row %width
%linear = OpIAdd %uint %x %row_offset
%base = OpIMul %uint %linear %four
%dynamic_dim = OpUMod %uint %x %three
'''
    for n, dim in enumerate(["zero", "one", "two", "dynamic_dim"]):
        offset = ["zero", "one", "two", "three"][n]
        assembly += f'''OpStore %arg %{dim}
%value_{n} = OpFunctionCall %uint %get_id %arg
%offset_{n} = OpIAdd %uint %base %{offset}
%dest_{n} = OpAccessChain %output_uint_ptr %output %zero %offset_{n}
OpStore %dest_{n} %value_{n}
'''
    return assembly + "OpReturn\nOpFunctionEnd\n"


def add_helper_input(assembly):
    """Include the library's WorkgroupId input in the caller's entry-point interface."""
    workgroup = re.search(r"OpDecorate (%\w+) BuiltIn WorkgroupId", assembly).group(1)
    lines = assembly.splitlines()
    for n, line in enumerate(lines):
        if "OpEntryPoint GLCompute" in line and workgroup not in line.split():
            lines[n] += " " + workgroup
    return "\n".join(lines) + "\n"


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("helper.comp").read_text()
    # Older glslang requires an entry point even when compiling reusable helpers.
    # Keep the helper uncalled, then remove only this empty entry point.
    (out / "helper_with_entry.comp").write_text(source + "\nvoid main() {}\n")
    run("glslangValidator", "-V", "--target-env", "vulkan1.1", "-Od", "--keep-uncalled",
        out / "helper_with_entry.comp", "-o", out / "helper.spv")
    helper = run("spirv-dis", out / "helper.spv")
    (out / "helper.spvasm").write_text(helper)
    (out / "library.spvasm").write_text(export_helper(helper))
    (out / "caller.spvasm").write_text(caller_assembly())
    for name in ["library", "caller"]:
        run("spirv-as", "--target-env", "spv1.3", out / f"{name}.spvasm", "-o", out / f"{name}.spv")
        run("spirv-val", "--target-env", "spv1.3", out / f"{name}.spv")
    run("spirv-link", "--target-env", "spv1.3", out / "caller.spv", out / "library.spv",
        "-o", out / "linked.spv")
    linked = run("spirv-dis", out / "linked.spv")
    (out / "linked.spvasm").write_text(linked)
    # Linking functions does not necessarily merge their input interfaces.
    (out / "kernel.spvasm").write_text(add_helper_input(linked))
    run("spirv-as", "--target-env", "spv1.3", out / "kernel.spvasm", "-o", out / "kernel.spv")
    run("spirv-val", "--target-env", "vulkan1.1", out / "kernel.spv")
    run("spirv-opt", "--target-env=vulkan1.1", "-O", out / "kernel.spv", "-o", out / "optimized.spv")
    run("spirv-val", "--target-env", "vulkan1.1", out / "optimized.spv")
    optimized = run("spirv-dis", out / "optimized.spv")
    (out / "optimized.spvasm").write_text(optimized)
    calls = len(re.findall(r"\bOpFunctionCall\b", optimized))
    if calls:
        raise RuntimeError(f"Helper calls remain after optimization: {calls}")
    report = {"function_calls_before": linked.count("OpFunctionCall"), "function_calls_after": calls,
              "constant_component_extracts": re.findall(r"OpCompositeExtract[^\n]+", optimized)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Execute both unoptimized and optimized kernels on Vulkan")
    args = parser.parse_args()
    build(args.out)
    if args.run:
        from run_vulkan import check_kernel

        for filename in ["kernel.spv", "optimized.spv"]:
            check_kernel(args.out / filename)
