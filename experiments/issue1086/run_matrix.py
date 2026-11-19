"""Run each math-mode cell in a new process, preserving complete logs and JSON records."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--arch", choices=("cpu", "cuda"), required=True)
parser.add_argument("--probe", choices=("reproduce", "genesis_probe"), default="reproduce")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--modes", nargs="+", default=["fast", "no_reassoc", "no_reassoc_no_contract", "strict"])
args, extra = parser.parse_known_args()
args.output.mkdir(parents=True, exist_ok=True)
records = []
failures = []
for precision in ("32", "64"):
    for mode in args.modes:
        label = f"{args.probe}_{args.arch}_{precision}_{mode}"
        env = os.environ.copy()
        env["QD_OFFLINE_CACHE"] = "0"
        if args.probe == "reproduce":
            dump_path = args.output / label
            dump_path.mkdir(exist_ok=True)
            env.update(QD_DUMP_IR="1", QD_DEBUG_DUMP_PATH=str(dump_path.resolve()))
        command = [
            sys.executable,
            str(Path(__file__).resolve().with_name(args.probe + ".py")),
            "--arch",
            args.arch,
            "--precision",
            precision,
            "--mode",
            mode,
            *extra,
        ]
        print("RUN " + json.dumps(command), flush=True)
        with (args.output / (label + ".log")).open("w") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=args.output.resolve()
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if line.startswith("RESULT "):
                    records.append(json.loads(line.removeprefix("RESULT ")))
            returncode = process.wait()
        if returncode:
            failures.append({"cell": label, "returncode": returncode})
        (args.output / "results.json").write_text(
            json.dumps({"results": records, "failures": failures}, indent=2) + "\n"
        )
print("MATRIX_COMPLETE " + json.dumps({"results": len(records), "failures": failures}), flush=True)
sys.exit(bool(failures))
