import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.quant_basic, debug=True)
def test_simple_array():
    qi13 = qd.types.quant.int(13, True)
    qu19 = qd.types.quant.int(19, False)

    x = qd.field(dtype=qi13)
    y = qd.field(dtype=qu19)

    N = 12

    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x, y)
    qd.root.dense(qd.i, N).place(bitpack)

    @qd.kernel
    def set_val():
        for i in range(N):
            x[i] = -(2**i)
            y[i] = 2**i - 1

    @qd.kernel
    def verify_val():
        for i in range(N):
            assert x[i] == -(2**i)
            assert y[i] == 2**i - 1

    set_val()
    verify_val()

    # Test read and write in Python-scope by calling the wrapped, untranslated function body
    set_val.__wrapped__()
    verify_val.__wrapped__()


# TODO: remove excluding of qd.metal
@test_utils.test(require=qd.extension.quant_basic, exclude=[qd.metal], debug=True)
def test_quant_int_load_and_store():
    qi13 = qd.types.quant.int(13, True)
    qu14 = qd.types.quant.int(14, False)
    qi5 = qd.types.quant.int(5, True)

    x = qd.field(dtype=qi13)
    y = qd.field(dtype=qu14)
    z = qd.field(dtype=qi5)

    test_case_np = np.array(
        [
            [2**12 - 1, 2**14 - 1, -(2**3)],
            [2**11 - 1, 2**13 - 1, -(2**2)],
            [0, 0, 0],
            [123, 4567, 8],
            [10, 31, 11],
        ],
        dtype=np.int32,
    )

    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x, y, z)
    qd.root.place(bitpack)
    test_case = qd.Vector.field(3, dtype=qd.i32, shape=len(test_case_np))
    test_case.from_numpy(test_case_np)

    @qd.kernel
    def set_val(idx: qd.i32):
        x[None] = test_case[idx][0]
        y[None] = test_case[idx][1]
        z[None] = test_case[idx][2]

    @qd.kernel
    def verify_val(idx: qd.i32):
        assert x[None] == test_case[idx][0]
        assert y[None] == test_case[idx][1]
        assert z[None] == test_case[idx][2]

    for idx in range(len(test_case_np)):
        set_val(idx)
        verify_val(idx)

    # Test read and write in Python-scope by calling the wrapped, untranslated function body
    for idx in range(len(test_case_np)):
        set_val.__wrapped__(idx)
        verify_val.__wrapped__(idx)


@test_utils.test(require=qd.extension.quant_basic)
def test_quant_int_full_struct():
    qit = qd.types.quant.int(32, True)
    x = qd.field(dtype=qit)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    qd.root.dense(qd.i, 1).place(bitpack)

    x[0] = 15
    assert x[0] == 15

    x[0] = 12
    assert x[0] == 12


def test_bitpacked_fields():
    def test_single_bitpacked_fields(physical_type, compute_type, quant_bits, test_case):
        qd.init(arch=qd.cpu, debug=True)

        qit1 = qd.types.quant.int(quant_bits[0], True, compute_type)
        qit2 = qd.types.quant.int(quant_bits[1], False, compute_type)
        qit3 = qd.types.quant.int(quant_bits[2], True, compute_type)

        a = qd.field(dtype=qit1)
        b = qd.field(dtype=qit2)
        c = qd.field(dtype=qit3)
        bitpack = qd.BitpackedFields(max_num_bits=physical_type)
        bitpack.place(a, b, c)
        qd.root.place(bitpack)

        @qd.kernel
        def set_val(test_val: qd.types.ndarray()):
            a[None] = test_val[0]
            b[None] = test_val[1]
            c[None] = test_val[2]

        @qd.kernel
        def verify_val(test_val: qd.types.ndarray()):
            assert a[None] == test_val[0]
            assert b[None] == test_val[1]
            assert c[None] == test_val[2]

        set_val(test_case)
        verify_val(test_case)

        qd.reset()

    test_single_bitpacked_fields(8, qd.i8, [3, 3, 2], np.array([2**2 - 1, 2**3 - 1, -(2**1)]))
    test_single_bitpacked_fields(16, qd.i16, [4, 7, 5], np.array([2**3 - 1, 2**7 - 1, -(2**4)]))
    test_single_bitpacked_fields(32, qd.i32, [17, 11, 4], np.array([2**16 - 1, 2**10 - 1, -(2**3)]))
    test_single_bitpacked_fields(64, qd.i64, [32, 23, 9], np.array([2**31 - 1, 2**23 - 1, -(2**8)]))
    test_single_bitpacked_fields(32, qd.i16, [7, 12, 13], np.array([2**6 - 1, 2**12 - 1, -(2**12)]))
    test_single_bitpacked_fields(64, qd.i32, [18, 22, 24], np.array([2**17 - 1, 2**22 - 1, -(2**23)]))

    test_single_bitpacked_fields(16, qd.i16, [5, 5, 6], np.array([15, 5, 20]))
    test_single_bitpacked_fields(32, qd.i32, [10, 10, 12], np.array([11, 19, 2020]))


@test_utils.test(require=[qd.extension.quant_basic, qd.extension.sparse], debug=True)
def test_bitpacked_fields_struct_for():
    block_size = 16
    N = 64
    cell = qd.root.pointer(qd.i, N // block_size)
    fixed32 = qd.types.quant.fixed(bits=32, max_value=1024)

    x = qd.field(dtype=fixed32)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    cell.dense(qd.i, block_size).place(bitpack)

    for i in range(N):
        if i // block_size % 2 == 0:
            x[i] = 0

    @qd.kernel
    def assign():
        for i in x:
            x[i] = qd.cast(i, float)

    assign()

    for i in range(N):
        if i // block_size % 2 == 0:
            assert x[i] == pytest.approx(i, abs=1e-3)
        else:
            assert x[i] == 0


@test_utils.test(require=qd.extension.quant_basic, debug=True)
def test_multiple_types():
    f15 = qd.types.quant.float(exp=5, frac=10)
    f18 = qd.types.quant.float(exp=5, frac=13)
    u4 = qd.types.quant.int(bits=4, signed=False)

    p = qd.field(dtype=f15)
    q = qd.field(dtype=f18)
    r = qd.field(dtype=u4)

    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(p, q, shared_exponent=True)
    bitpack.place(r)
    qd.root.dense(qd.i, 12).place(bitpack)

    @qd.kernel
    def set_val():
        for i in p:
            p[i] = i * 3
            q[i] = i * 2
            r[i] = i

    @qd.kernel
    def verify_val():
        for i in p:
            assert p[i] == i * 3
            assert q[i] == i * 2
            assert r[i] == i

    set_val()
    verify_val()


@test_utils.test()
def test_invalid_place():
    f15 = qd.types.quant.float(exp=5, frac=10)
    p = qd.field(dtype=f15)
    bitpack = qd.BitpackedFields(max_num_bits=32)
    with pytest.raises(
        qd.QuadrantsCompilationError,
        match="At least 2 fields need to be placed when shared_exponent=True",
    ):
        bitpack.place(p, shared_exponent=True)
