"""Compile a GLSL helper separately, link a generated caller, and inspect the result.

This deliberately exercises module integration without modifying Quadrants' compiler.
The caller matches this helper's GLSL calling convention, not arbitrary shader functions.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.check_output(list(map(str, args)), text=True)


def caller_assembly():
    """Generate a caller that imports the helper but never reads WorkgroupId itself.

    Dispatch is (7, 3, 2), local size is (4, 1, 1). Each invocation writes four
    values: helper(0), helper(1), helper(2), and helper(global_x % 3).
    GLSL uses a Function-storage pointer for this scalar parameter.
    """
    assembly = """OpCapability Shader
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
"""
    for n, dim in enumerate(["zero", "one", "two", "dynamic_dim"]):
        offset = ["zero", "one", "two", "three"][n]
        assembly += f"""OpStore %arg %{dim}
%value_{n} = OpFunctionCall %uint %get_id %arg
%offset_{n} = OpIAdd %uint %base %{offset}
%dest_{n} = OpAccessChain %output_uint_ptr %output %zero %offset_{n}
OpStore %dest_{n} %value_{n}
"""
    return assembly + "OpReturn\nOpFunctionEnd\n"


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    run(
        "glslangValidator",
        "-V",
        "--target-env",
        "vulkan1.1",
        "-Od",
        "--no-link",
        Path(__file__).with_name("helper.comp"),
        "-o",
        out / "library.spv",
    )
    library = run("spirv-dis", out / "library.spv")
    (out / "library.spvasm").write_text(library)
    run("spirv-val", "--target-env", "spv1.3", out / "library.spv")
    (out / "caller.spvasm").write_text(caller_assembly())
    run("spirv-as", "--target-env", "spv1.3", out / "caller.spvasm", "-o", out / "caller.spv")
    run("spirv-val", "--target-env", "spv1.3", out / "caller.spv")
    run("spirv-link", "--target-env", "spv1.3", out / "caller.spv", out / "library.spv", "-o", out / "kernel.spv")
    linked = run("spirv-dis", out / "kernel.spv")
    (out / "linked.spvasm").write_text(linked)
    workgroup = re.search(r"OpDecorate (%\w+) BuiltIn WorkgroupId", linked).group(1)
    entry = next(line for line in linked.splitlines() if "OpEntryPoint GLCompute" in line)
    if workgroup not in entry.split():
        raise RuntimeError("Linker did not include the helper's WorkgroupId in the entry-point interface")
    run("spirv-val", "--target-env", "vulkan1.1", out / "kernel.spv")
    run("spirv-opt", "--target-env=vulkan1.1", "-O", out / "kernel.spv", "-o", out / "optimized.spv")
    run("spirv-val", "--target-env", "vulkan1.1", out / "optimized.spv")
    optimized = run("spirv-dis", out / "optimized.spv")
    (out / "optimized.spvasm").write_text(optimized)
    calls = len(re.findall(r"\bOpFunctionCall\b", optimized))
    if calls:
        raise RuntimeError(f"Helper calls remain after optimization: {calls}")
    workgroup = re.search(r"OpDecorate (%\w+) BuiltIn WorkgroupId", optimized).group(1)
    accesses = re.findall(rf"OpAccessChain %\w+ {re.escape(workgroup)} (%\w+)", optimized)
    constants = dict(re.findall(r"(%\w+) = OpConstant %\w+ (\d+)", optimized))
    constant_dimensions = [int(constants[index]) for index in accesses if index in constants]
    if constant_dimensions != [0, 1, 2] or len(accesses) != 4:
        raise RuntimeError(f"Expected three constant component reads and one dynamic read, got {accesses}")
    report = {
        "function_calls_before": linked.count("OpFunctionCall"),
        "function_calls_after": calls,
        "constant_workgroup_dimensions": constant_dimensions,
        "dynamic_workgroup_reads": len(accesses) - len(constant_dimensions),
        "linker_included_helper_input": True,
    }
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
