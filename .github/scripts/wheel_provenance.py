"""Attach the source tree and wheel digest to the wheel artifact."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
for wheel in Path("wheelhouse").glob("*.whl"):
    wheel.with_suffix(".json").write_text(
        json.dumps(
            {
                "commit": commit,
                "tree": tree,
                "run_id": int(os.environ["GITHUB_RUN_ID"]),
                "attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
                "wheel": wheel.name,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n"
    )
