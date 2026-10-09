"""Exercise the demo's actual resident dense/CSR, update, and readout composition."""

import numpy as np
import pytest

import quadrants as qd
from misc.demos.csr_spmm_esn import Config, compare, device, fit, fixture, oracle, run

from tests import test_utils


def test_csr_spmm_esn_demo_workspace_guard():
    # Dense-degree CSR needs its own budget alongside dense weights and oracle copies.
    with pytest.raises(MemoryError, match="2 GiB"):
        Config(n=2048, batch=16, degree=2048, washout=0, train=4, test=2, free=2).validate()
    # The documented independent-trial demonstration remains within the bound.
    Config(n=512, batch=32, degree=16, washout=100, train=1000, test=300, free=128).validate()


@test_utils.test(arch=[qd.cpu, qd.metal], fast_math=False)
def test_csr_spmm_esn_demo_composition():
    # Two independently weighted trials, with a partial grid and both ping-pong parities.
    c = Config(n=7, batch=2, degree=3, washout=7, train=24, test=20, free=16)
    arrays, _ = fixture(c)
    before = {key: value.copy() for key, value in arrays.items()}
    assert not np.array_equal(arrays["dense"][0], arrays["dense"][1])
    columns = arrays["col_idx"].reshape(c.batch, c.n, c.degree)
    assert np.all(np.diff(columns, axis=2) > 0)
    for trial in range(c.batch):
        assert np.all((trial * c.n <= columns[trial]) & (columns[trial] < (trial + 1) * c.n))
    expected, host_evaluate = oracle(c, arrays)
    shared = fit(c, arrays, expected["states"], c.train)
    expected.update(host_evaluate(shared))
    paths = []
    for dense in (True, False):
        actual, evaluate, uploaded = device(c, arrays, dense)
        actual.update(evaluate(shared))
        for key, atol, rtol in (
            ("linear", 3e-5, 3e-4),
            ("states", 3e-5, 3e-4),
            ("predictions", 1e-4, 1e-3),
            ("free", 3e-4, 3e-3),
        ):
            assert compare(actual[key], expected[key], atol, rtol)["passed"], key
        # Refit this path's own states, then restore shared coefficients. This detects
        # stale coefficient bindings and a free run that fails to reseed from training.
        own = evaluate(fit(c, arrays, actual["states"], c.train // 2))
        assert np.isfinite(own["free"]).all()
        assert not np.array_equal(own["predictions"], actual["predictions"])
        restored = evaluate(shared)
        for key in restored:
            np.testing.assert_array_equal(restored[key], actual[key])
        for key, buffer in uploaded.items():
            expected_input = before[key].transpose(0, 2, 1) if key == "dense" else before[key]
            np.testing.assert_array_equal(buffer.to_numpy(), expected_input)
        paths.append(actual)
    for key in paths[0]:
        np.testing.assert_allclose(paths[0][key], paths[1][key], atol=3e-4, rtol=3e-3)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])


@test_utils.test(arch=[qd.cpu, qd.metal], fast_math=False)
@pytest.mark.parametrize("train", [24, 240])
def test_csr_spmm_esn_demo_timing(train):
    # Odd short rollouts and the 256-step cap exercise actual last-buffer ownership.
    c = Config(n=7, batch=2, degree=3, washout=7, train=train, test=20, free=16)
    report, _ = run(c, timing_blocks=3, timing_warmup=1)
    assert report["results"]["passed"]
    timing = report["results"]["timing"]
    assert timing["resident_steps"] == min(256, c.steps)
    for scope in ("resident", "workflow"):
        result = timing[scope]
        assert result["order_by_block"] == [["dense", "csr"], ["csr", "dense"], ["dense", "csr"]]
        for path in ("dense", "csr"):
            assert len(result[path]["seconds"]) == 3
            assert len(result["checks"][path]) == 4
            assert all(check["finite_positive_seconds"] for check in result["checks"][path])
            assert all(output["passed"] for check in result["checks"][path] for output in check["outputs"].values())
