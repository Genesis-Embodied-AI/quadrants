import pytest

import quadrants as ti

from tests import test_utils


@test_utils.test(require=ti.extension.quant_basic)
def test_quant_int_implicit_cast():
    qi13 = ti.types.quant.int(13, True)
    x = ti.field(dtype=qi13)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    ti.root.place(bitpack)

    @ti.kernel
    def foo():
        x[None] = 10.3

    foo()
    assert x[None] == 10


@test_utils.test(
    require=ti.extension.quant_basic,
)
def test_quant_store_fusion() -> None:
    x = ti.field(dtype=ti.types.quant.int(16, True))
    y = ti.field(dtype=ti.types.quant.int(16, True))
    v = ti.BitpackedFields(max_num_bits=32)
    v.place(x, y)
    ti.root.dense(ti.i, 10).place(v)

    z = ti.field(dtype=ti.i32, shape=(10, 2))

    @ti.real_func
    def probe(x: ti.template(), z: ti.template(), i: int, j: int) -> None:
        z[i, j] = x[i]

    # should fuse store
    # note: don't think this actually tests that store is fused?
    @ti.kernel
    def store():
        ti.loop_config(serialize=True)
        for i in range(10):
            x[i] = i
            y[i] = i + 1
            probe(x, z, i, 0)
            probe(y, z, i, 1)

    store()
    ti.sync()

    print("z", z.to_numpy())

    for i in range(10):
        assert z[i, 0] == i
        assert z[i, 1] == i + 1
        assert x[i] == i
        assert y[i] == i + 1


@pytest.mark.xfail(
    reason="Bug in store fusion. TODO: fix this. Logged at https://linear.app/genesis-ai-company/issue/CMP-57/fuse-store-bug-for-16-bit-quantization"
)
@test_utils.test(
    require=ti.extension.quant_basic,
)
def test_quant_store_no_fusion() -> None:
    x = ti.field(dtype=ti.types.quant.int(16, True))
    y = ti.field(dtype=ti.types.quant.int(16, True))
    v = ti.BitpackedFields(max_num_bits=32)
    v.place(x, y)
    ti.root.dense(ti.i, 10).place(v)

    z = ti.field(dtype=ti.i32, shape=(10, 2))

    @ti.real_func
    def probe(x: ti.template(), z: ti.template(), i: int, j: int) -> None:
        z[i, j] = x[i]

    @ti.kernel
    def store():
        ti.loop_config(serialize=True)
        for i in range(10):
            x[i] = i
            probe(x, z, i, 0)
            y[i] = i + 1
            probe(y, z, i, 1)

    store()
    ti.sync()

    print("z", z.to_numpy())

    for i in range(10):
        assert z[i, 0] == i
        assert z[i, 1] == i + 1
        assert x[i] == i
        assert y[i] == i + 1
