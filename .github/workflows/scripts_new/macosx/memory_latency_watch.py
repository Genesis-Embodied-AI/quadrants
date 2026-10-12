"""Same-VM fixed-memory sidecar: no Quadrants import, no GPU runtime, one private 64-MiB graph.

The matching worker probe runs after its direct GC-list scan. This independent process samples every five seconds.
The added memory and cache traffic are diagnostic observer effects, not part of an uninstrumented suite benchmark.
"""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def memory_watch(python, folder, env):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "sidecar.log").open("w") as stream:
        process = subprocess.Popen(
            [str(python), __file__, str(folder)], env=env, stdout=stream, stderr=subprocess.STDOUT
        )
        print("Started independent memory control", process.pid, flush=True)
        try:
            yield
        finally:
            already_exited = process.poll() is not None
            if not already_exited:
                process.terminate()
            try:
                code = process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            (folder / "sidecar-status.json").write_text(
                json.dumps(dict(returncode=code, exited_before_suite=already_exited)) + "\n"
            )
            if code or already_exited:
                raise RuntimeError(f"Independent memory control exited unexpectedly: {code}, early={already_exited}")


def main():
    import _gc_walk

    assert sys.platform == "darwin" and sys.version_info[:2] == (3, 10)
    assert "quadrants" not in sys.modules
    folder = Path(sys.argv[1])
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    with (folder / "memory-sidecar.jsonl").open("w", buffering=1) as stream:
        stream.write(
            json.dumps(dict(kind="start", timestamp=time.time(), pid=os.getpid(), quadrants_imported=False)) + "\n"
        )
        while running:
            start = time.time()
            result = _gc_walk.memory()
            stream.write(json.dumps(dict(kind="memory", timestamp=time.time(), started_at=start, **result)) + "\n")
            time.sleep(5)
        assert "quadrants" not in sys.modules
        stream.write(json.dumps(dict(kind="end", timestamp=time.time(), quadrants_imported=False)) + "\n")


if __name__ == "__main__":
    main()
