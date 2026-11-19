"""Print compact summaries of the machine-readable experiment results."""

import json
import sys
from pathlib import Path

for argument in sys.argv[1:]:
    data = json.loads(Path(argument).read_text())
    print(argument)
    for result in data["results"]:
        label = f'{result["arch"]} fp{result["precision"]} {result["mode"]}'
        if "differing_rows" in result:
            print(label, "different matrices:", result["differing_rows"], "max_abs:", result["max_abs"])
        else:
            restore = [name for name, value in result["restore"].items() if not value["bitwise_equal"]]
            replay = [
                step + 1
                for step, fields in enumerate(result["replay"])
                if any(not value["bitwise_equal"] for value in fields.values())
            ]
            max_qpos_error = max(fields["qpos"]["max_abs"] for fields in result["replay"])
            print(
                label, "restore differences:", restore, "different replay steps:", replay, "max qpos:", max_qpos_error
            )
            if "step_seconds" in result:
                print("  step milliseconds:", [round(value * 1000, 6) for value in result["step_seconds"]])
    print("failures:", data["failures"])
