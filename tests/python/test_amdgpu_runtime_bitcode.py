"""Prepared runtime bytes survive reset; mutable LLVM modules and field layouts do not."""

import quadrants as qd

from tests.helpers import test_utils


@test_utils.test(arch=qd.amdgpu, offline_cache=False)
def test_runtime_bitcode_fresh_structures(capfd):
    for cycle, size in enumerate((9, 31, 6, 17, 9, 31)):
        qd.reset()
        capfd.readouterr()
        qd.init(
            arch=qd.amdgpu,
            offline_cache=True,
            fast_math=bool(cycle % 2),
            log_level=qd.TRACE,
        )
        output = "".join(capfd.readouterr())
        if cycle:
            assert "AMDGPU runtime bitcode cache hit" in output
        values = qd.Vector.field(2 if cycle % 2 else 3, qd.f32, shape=size)

        @qd.kernel
        def fill():
            for i in values:
                for j in qd.static(range(values.n)):
                    values[i][j] = i * 10 + j

        fill()
        assert values[size - 1][values.n - 1] == (size - 1) * 10 + values.n - 1
    qd.reset()
