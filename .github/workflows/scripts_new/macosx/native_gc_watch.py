"""Capture bounded native diagnostics while a worker is inside a long full GC."""

import json
import subprocess
import threading
import time
from pathlib import Path


class NativeGCWatch:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.stop = threading.Event()
        self.counts = {}
        self.seen = set()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=30)

    def capture(self, command, destination):
        with destination.open("w") as stream:
            try:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=15)
                return result.returncode
            except subprocess.TimeoutExpired:
                return "timeout"

    def watch(self):
        while not self.stop.wait(1):
            for path in self.folder.glob("gc-*.jsonl"):
                pid = int(path.stem.rsplit("-", 1)[1])
                if self.counts.get(pid, 0) >= 3:
                    continue
                try:
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 8192))
                        lines = stream.read().splitlines()
                    record = json.loads(lines[-1])
                except (OSError, ValueError, IndexError):
                    continue
                stamp = record.get("timestamp", 0)
                if record["kind"] != "gc_begin" or time.time() - stamp < 3 or (pid, stamp) in self.seen:
                    continue
                self.seen.add((pid, stamp))
                index = self.counts.get(pid, 0)
                self.counts[pid] = index + 1
                folder = self.folder / f"native-{pid}-{index}"
                folder.mkdir()
                metadata = dict(trigger=record, capture_timestamp=time.time(), commands={})
                commands = {
                    "sample": ["/usr/bin/sample", str(pid), "1", "1", "-file", str(folder / "sample.txt")],
                    "vm_stat": ["/usr/bin/vm_stat"],
                    "swap": ["/usr/sbin/sysctl", "vm.swapusage"],
                    "process": ["/bin/ps", "-p", str(pid), "-o", "pid,ppid,%cpu,rss,vsz,state,time,command"],
                    "threads": ["/bin/ps", "-M", "-p", str(pid)],
                    "system-processes": ["/bin/ps", "-axo", "pid,ppid,%cpu,rss,state,time,comm"],
                    "system-threads": ["/bin/ps", "-eM"],
                }
                for name, command in commands.items():
                    metadata["commands"][name] = self.capture(command, folder / f"{name}-command.txt")
                (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
