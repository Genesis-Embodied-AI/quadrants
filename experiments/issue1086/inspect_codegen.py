"""Count reassociation flags and fused instructions in the reproduction's one_loop modules."""

import argparse
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cpu-dir", type=Path, required=True)
parser.add_argument("--cuda-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
records = []
for arch, root in (("cpu", args.cpu_dir), ("cuda", args.cuda_dir)):
    for precision in ("32", "64"):
        for mode in ("fast", "no_reassoc", "no_reassoc_no_contract", "strict"):
            directory = root / f"reproduce_{arch}_{precision}_{mode}"
            (ir,) = directory.glob("one_loop*_llvm.ll")
            if arch == "cpu":
                (assembly,) = [path for path in directory.glob("*.s") if "one_loop" in path.read_text()]
                fused_pattern = r"\bv(?:fmadd|fmsub|fnmadd|fnmsub)\w*\b"
            else:
                (assembly,) = directory.glob("one_loop*.ptx")
                fused_pattern = r"\bfma\."
            records.append(
                {
                    "arch": arch,
                    "precision": precision,
                    "mode": mode,
                    "reassoc_occurrences": len(re.findall(r"\breassoc\b", ir.read_text())),
                    "fused_instructions": len(re.findall(fused_pattern, assembly.read_text())),
                    "ir_file": str(ir.relative_to(root)),
                    "assembly_file": str(assembly.relative_to(root)),
                }
            )
args.output.write_text(json.dumps(records, indent=2) + "\n")
for record in records:
    print(record)
