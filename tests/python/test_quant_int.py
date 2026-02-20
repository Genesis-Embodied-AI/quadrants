import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.quant_basic)
def test_quant_int_implicit_cast():
    qi13 = qd.types.quant.int(13, True)
    x = qd.field(dtype=qi13)

    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    qd.root.place(bitpack)

    @qd.kernel
    def foo():
        x[None] = 10.3

    foo()
    assert x[None] == 10


@test_utils.test(
    require=qd.extension.quant_basic,
)
def test_quant_store_fusion() -> None:
    x = qd.field(dtype=qd.types.quant.int(16, True))
    y = qd.field(dtype=qd.types.quant.int(16, True))
    v = qd.BitpackedFields(max_num_bits=32)
    v.place(x, y)
    qd.root.dense(qd.i, 10).place(v)

    z = qd.field(dtype=qd.i32, shape=(10, 2))

    @qd.real_func
    def probe(x: qd.template(), z: qd.template(), i: int, j: int) -> None:
        z[i, j] = x[i]

    # should fuse store
    # note: don't think this actually tests that store is fused?
    @qd.kernel
    def store():
        qd.loop_config(serialize=True)
        for i in range(10):
            x[i] = i
            y[i] = i + 1
            probe(x, z, i, 0)
            probe(y, z, i, 1)

    store()
    qd.sync()

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
    require=qd.extension.quant_basic,
)
def test_quant_store_no_fusion() -> None:
    x = qd.field(dtype=qd.types.quant.int(16, True))
    y = qd.field(dtype=qd.types.quant.int(16, True))
    v = qd.BitpackedFields(max_num_bits=32)
    v.place(x, y)
    qd.root.dense(qd.i, 10).place(v)

    z = qd.field(dtype=qd.i32, shape=(10, 2))

    @qd.real_func
    def probe(x: qd.template(), z: qd.template(), i: int, j: int) -> None:
        z[i, j] = x[i]

    @qd.kernel
    def store():
        qd.loop_config(serialize=True)
        for i in range(10):
            x[i] = i
            probe(x, z, i, 0)
            y[i] = i + 1
            probe(y, z, i, 1)

    store()
    qd.sync()

    print("z", z.to_numpy())

    for i in range(10):
        assert z[i, 0] == i
        assert z[i, 1] == i + 1
        assert x[i] == i
        assert y[i] == i + 1
