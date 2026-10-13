#!/bin/bash
# Three Metal workers plus one CPU reader that never calls Metal.
# mode 1 is the placement-heap case. mode 0 is the ordinary-buffer control.
# Usage: ./run.sh <mode> <cycles> <workers>
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-1}"
CYCLES="${2:-1500}"
WORKERS="${3:-3}"
OUT="${OUT:-output}"
mkdir -p "$OUT"

xcrun clang -std=c11 -O2 -Wall -Wextra -Werror cpu_read.c -o "$OUT/cpu_read"
xcrun clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wextra -Werror \
    -framework Foundation -framework Metal heap_cycle.mm -o "$OUT/heap_cycle"
{
    sw_vers
    xcrun clang++ --version
    echo "mode=$MODE cycles=$CYCLES workers=$WORKERS"
} > "$OUT/environment.txt"

"$OUT/cpu_read" > "$OUT/cpu_read.txt" 2>&1 &
READER=$!
sleep 6

python3 -c 'import time; print(f"{time.time():.3f}")' > "$OUT/workers_started_at.txt"
WORKER_PIDS=()
for i in $(seq 1 "$WORKERS"); do
    "$OUT/heap_cycle" "$MODE" "$CYCLES" > "$OUT/worker-$i.txt" 2>&1 &
    WORKER_PIDS+=($!)
done
echo "reader=$READER workers=${WORKER_PIDS[*]}" | tee "$OUT/pids.txt"

FAIL=0
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "$pid"; then FAIL=1; fi
done
python3 -c 'import time; print(f"{time.time():.3f}")' > "$OUT/workers_exited_at.txt"
# Keep sampling after the workers exit. The slow reads drop once those processes are gone.
sleep 30
kill "$READER" || true
wait "$READER" || true

python3 - "$OUT" <<'PY'
import pathlib, sys
out = pathlib.Path(sys.argv[1])
started = float((out / "workers_started_at.txt").read_text())
exited = float((out / "workers_exited_at.txt").read_text())
during, after = [], []
for line in (out / "cpu_read.txt").read_text().splitlines():
    if not line.startswith("sample="):
        continue
    fields = dict(part.split("=", 1) for part in line.split())
    row = (float(fields["epoch"]), float(fields["wall1"]), float(fields["wall2"]), float(fields["cpu1"]),
           int(fields["resident"]), int(fields["pages"]), int(fields["minor"]), int(fields["major"]))
    if row[0] < started:
        continue
    (during if row[0] <= exited else after).append(row)
if not during or not after:
    print(f"VERDICT=TOO_FEW_SAMPLES during={len(during)} after={len(after)}")
    sys.exit(2)
d = max(during, key=lambda row: row[1])
a = max(after, key=lambda row: row[1])
print(f"during_max_wall1={d[1]:.3f} pair_wall2={d[2]:.3f} cpu1={d[3]:.3f} resident={d[4]}/{d[5]} minor={d[6]} major={d[7]}")
print(f"after_max_wall1={a[1]:.3f} pair_wall2={a[2]:.3f} cpu1={a[3]:.3f}")
# A hit is a multi-second first walk while the immediate repeat stays under a third of a second.
hit = d[1] >= 1.0 and d[2] < 0.3
print("VERDICT=" + ("STALL" if hit else "NO_STALL"))
sys.exit(0)
PY
exit "$FAIL"
