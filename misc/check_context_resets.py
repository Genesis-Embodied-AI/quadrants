"""Inspect repeated field compilation without pytest output capture."""

import quadrants as qd

for cycle in range(3):
    print(f"CYCLE {cycle}", flush=True)
    qd.init(arch=qd.amdgpu, offline_cache=True, log_level=qd.TRACE)
    values = qd.field(qd.i32, shape=32)

    @qd.kernel
    def fill():
        for i in values:
            values[i] = i

    fill()
    assert values[17] == 17
    qd.reset()
