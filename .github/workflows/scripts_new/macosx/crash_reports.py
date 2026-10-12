"""Preserve new Python/pytest and memory-kill diagnostic reports from this ephemeral Mac runner."""

import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def crash_reports(output):
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        records = []
        for root in (Path.home() / "Library/Logs/DiagnosticReports", Path("/Library/Logs/DiagnosticReports")):
            try:
                paths = list(root.rglob("*.ips")) + list(root.rglob("*.crash"))
                for path in paths:
                    if not path.name.lower().startswith(("python", "pytest", "jetsamevent")):
                        continue
                    if path.stat().st_mtime < started:
                        continue
                    destination = output / f"{len(records):03d}-{path.name}.txt"
                    shutil.copy2(path, destination)
                    records.append(dict(source=str(path), saved=destination.name))
            except OSError as error:
                records.append(dict(source=str(root), error=str(error)))
        (output / "manifest.json").write_text(json.dumps(dict(started_at=started, reports=records), indent=2) + "\n")
