import pathlib

import pytest
import quadrants as qd

from tests import test_utils


@test_utils.test(offline_cache=False)
def test_loop_config_name_ir_dump(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("QD_DUMP_IR", "1")
    qd.lang.impl.current_cfg().debug_dump_path = str(tmp_path)

    n = 128
    a = qd.ndarray(qd.i32, shape=(n,))
    b = qd.ndarray(qd.i32, shape=(n,))
    c = qd.ndarray(qd.i32, shape=(n,))

    @qd.kernel
    def three_loops(
        a: qd.types.ndarray(dtype=qd.i32, ndim=1),
        b: qd.types.ndarray(dtype=qd.i32, ndim=1),
        c: qd.types.ndarray(dtype=qd.i32, ndim=1),
    ):
        qd.loop_config(name="fill_a")
        for i in range(n):
            a[i] = i

        qd.loop_config(name="fill_b")
        for i in range(n):
            b[i] = i * 2

        # Non-empty loop_config but no `name`: must not pick up fill_a/fill_b in IR.
        qd.loop_config(block_dim=64)
        for i in range(n):
            c[i] = i + 7

    three_loops(a, b, c)
    qd.sync()

    a_np = a.to_numpy()
    b_np = b.to_numpy()
    c_np = c.to_numpy()
    for i in range(n):
        assert a_np[i] == i
        assert b_np[i] == i * 2
        assert c_np[i] == i + 7

    offload_files = list(tmp_path.glob("*after_offload*"))
    assert len(offload_files) > 0, f"No after_offload IR dumps found in {tmp_path}"
    combined = "\n".join(f.read_text() for f in offload_files)
    assert "loop_name=fill_a" in combined, f"loop_name=fill_a not found in IR dump:\n{combined}"
    assert "loop_name=fill_b" in combined, f"loop_name=fill_b not found in IR dump:\n{combined}"
    assert combined.count("loop_name=fill_a") == 1
    assert combined.count("loop_name=fill_b") == 1
    assert (
        combined.count("loop_name=") == 2
    ), f"Expected exactly two loop_name= annotations (named loops only), got {combined.count('loop_name=')}:\n{combined}"


@test_utils.test()
def test_loop_config_name_with_serialize():
    @qd.kernel
    def serial_named() -> qd.i32:
        a = 0
        qd.loop_config(serialize=True, name="serial_sum")
        for i in range(100):
            a += i
            if i == 10:
                break
        return a

    assert serial_named() == 55
