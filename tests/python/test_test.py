"""
This file tests if Quadrants's testing utilities are functional.

TODO: Skips these tests after all tests are using @qd.test
"""

import pytest

import quadrants as qd

from tests import test_utils

### `qd.test`


@test_utils.test()
def test_all_archs():
    assert qd.lang.impl.current_cfg().arch in test_utils.expected_archs()


@test_utils.test(arch=qd.cpu)
def test_arch_cpu():
    assert qd.lang.impl.current_cfg().arch in [qd.cpu]


@test_utils.test(arch=[qd.cpu])
def test_arch_list_cpu():
    assert qd.lang.impl.current_cfg().arch in [qd.cpu]


@test_utils.test(exclude=qd.cpu)
def test_exclude_cpu():
    assert qd.lang.impl.current_cfg().arch not in [qd.cpu]


@test_utils.test(exclude=[qd.cpu])
def test_exclude_list_cpu():
    assert qd.lang.impl.current_cfg().arch not in [qd.cpu]


@test_utils.test(arch=[qd.cpu, qd.metal])
def test_multiple_archs():
    assert qd.lang.impl.current_cfg().arch in [qd.cpu, qd.metal]


@test_utils.test(arch=qd.cpu, debug=True, advanced_optimization=False)
def test_init_args():
    assert qd.lang.impl.current_cfg().debug == True
    assert qd.lang.impl.current_cfg().advanced_optimization == False


@test_utils.test(require=qd.extension.sparse)
def test_require_extensions_1():
    assert qd.lang.impl.current_cfg().arch in [qd.cpu, qd.cuda, qd.metal]


@test_utils.test(arch=[qd.cpu], require=qd.extension.sparse)
def test_require_extensions_2():
    assert qd.lang.impl.current_cfg().arch in [qd.cpu]


@test_utils.test(arch=[qd.cpu], require=[qd.extension.sparse, qd.extension.bls])
def test_require_extensions_2():
    assert qd.lang.impl.current_cfg().arch in [qd.cuda]


### `test_utils.approx` and `test_utils.allclose`


@pytest.mark.parametrize("x", [0.1, 3])
@pytest.mark.parametrize("allclose", [test_utils.allclose, lambda x, y: x == test_utils.approx(y)])
@test_utils.test()
def test_allclose_rel(x, allclose):
    rel = test_utils.get_rel_eps()
    assert not allclose(x + x * rel * 3.0, x)
    assert not allclose(x + x * rel * 1.2, x)
    assert allclose(x + x * rel * 0.9, x)
    assert allclose(x + x * rel * 0.5, x)
    assert allclose(x, x)
    assert allclose(x - x * rel * 0.5, x)
    assert allclose(x - x * rel * 0.9, x)
    assert not allclose(x - x * rel * 1.2, x)
    assert not allclose(x - x * rel * 3.0, x)


@pytest.mark.parametrize("x", [0.1, 3])
@pytest.mark.parametrize("allclose", [test_utils.allclose, lambda x, y: x == test_utils.approx(y)])
@test_utils.test()
def test_allclose_rel_reordered1(x, allclose):
    rel = test_utils.get_rel_eps()
    assert not allclose(x + x * rel * 3.0, x)
    assert not allclose(x + x * rel * 1.2, x)
    assert allclose(x + x * rel * 0.9, x)
    assert allclose(x + x * rel * 0.5, x)
    assert allclose(x, x)
    assert allclose(x - x * rel * 0.5, x)
    assert allclose(x - x * rel * 0.9, x)
    assert not allclose(x - x * rel * 1.2, x)
    assert not allclose(x - x * rel * 3.0, x)


@pytest.mark.parametrize("x", [0.1, 3])
@pytest.mark.parametrize("allclose", [test_utils.allclose, lambda x, y: x == test_utils.approx(y)])
@test_utils.test()
def test_allclose_rel_reordered2(x, allclose):
    rel = test_utils.get_rel_eps()
    assert not allclose(x + x * rel * 3.0, x)
    assert not allclose(x + x * rel * 1.2, x)
    assert allclose(x + x * rel * 0.9, x)
    assert allclose(x + x * rel * 0.5, x)
    assert allclose(x, x)
    assert allclose(x - x * rel * 0.5, x)
    assert allclose(x - x * rel * 0.9, x)
    assert not allclose(x - x * rel * 1.2, x)
    assert not allclose(x - x * rel * 3.0, x)


@pytest.mark.skipif(qd._lib.core.with_metal(), reason="Skip metal because metal is used as the example")
def test_disable_fallback():
    with pytest.raises(RuntimeError):
        qd.init(arch=qd.metal, enable_fallback=False)
        qd.reset()
