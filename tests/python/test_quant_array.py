import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.quant, debug=True)
def test_1D_quant_array():
    qu1 = qd.types.quant.int(1, False)

    x = qd.field(dtype=qu1)

    N = 32

    qd.root.quant_array(qd.i, N, max_num_bits=32).place(x)

    @qd.kernel
    def set_val():
        for i in range(N):
            x[i] = i % 2

    @qd.kernel
    def verify_val():
        for i in range(N):
            assert x[i] == i % 2

    set_val()
    verify_val()


@test_utils.test(require=qd.extension.quant, debug=True)
def test_1D_quant_array_negative():
    N = 4
    qi7 = qd.types.quant.int(7)
    x = qd.field(dtype=qi7)
    qd.root.quant_array(qd.i, N, max_num_bits=32).place(x)

    @qd.kernel
    def assign():
        for i in range(N):
            assert x[i] == 0
            x[i] = -i
            assert x[i] == -i

    assign()


@test_utils.test(require=qd.extension.quant, debug=True)
def test_1D_quant_array_fixed():
    qfxt = qd.types.quant.fixed(bits=8, max_value=2)

    x = qd.field(dtype=qfxt)

    N = 4

    qd.root.quant_array(qd.i, N, max_num_bits=32).place(x)

    @qd.kernel
    def set_val():
        for i in range(N):
            x[i] = i * 0.5

    @qd.kernel
    def verify_val():
        for i in range(N):
            assert x[i] == i * 0.5

    set_val()
    verify_val()


@test_utils.test(require=qd.extension.quant, debug=True)
def test_2D_quant_array():
    qu1 = qd.types.quant.int(1, False)

    x = qd.field(dtype=qu1)

    M, N = 4, 8

    qd.root.quant_array(qd.ij, (M, N), max_num_bits=32).place(x)

    @qd.kernel
    def set_val():
        for i in range(M):
            for j in range(N):
                x[i, j] = (i * N + j) % 2

    @qd.kernel
    def verify_val():
        for i in range(M):
            for j in range(N):
                assert x[i, j] == (i * N + j) % 2

    set_val()
    verify_val()


@test_utils.test(require=qd.extension.quant, debug=True)
def test_quant_array_struct_for():
    block_size = 16
    N = 64
    cell = qd.root.pointer(qd.i, N // block_size)
    qi7 = qd.types.quant.int(7)

    x = qd.field(dtype=qi7)
    cell.dense(qd.i, block_size // 4).quant_array(qd.i, 4, max_num_bits=32).place(x)

    @qd.kernel
    def activate():
        for i in range(N):
            if i // block_size % 2 == 0:
                x[i] = i

    @qd.kernel
    def assign():
        for i in x:
            x[i] -= 1

    @qd.kernel
    def verify():
        for i in range(N):
            if i // block_size % 2 == 0:
                assert x[i] == i - 1
            else:
                assert x[i] == 0

    activate()
    assign()
    verify()
