import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.quant, debug=True, cfg_optimization=False)
def test_vectorized_struct_for():
    qu1 = qd.types.quant.int(1, False)

    x = qd.field(dtype=qu1)
    y = qd.field(dtype=qu1)

    N = 4096
    n_blocks = 4
    bits = 32
    boundary_offset = 1024

    block = qd.root.pointer(qd.ij, (n_blocks, n_blocks))
    block.dense(qd.ij, (N // n_blocks, N // (bits * n_blocks))).quant_array(qd.j, bits, max_num_bits=bits).place(x)
    block.dense(qd.ij, (N // n_blocks, N // (bits * n_blocks))).quant_array(qd.j, bits, max_num_bits=bits).place(y)

    @qd.kernel
    def init():
        for i, j in qd.ndrange(
            (boundary_offset, N - boundary_offset),
            (boundary_offset, N - boundary_offset),
        ):
            x[i, j] = qd.random(dtype=qd.i32) % 2

    @qd.kernel
    def assign_vectorized():
        qd.loop_config(bit_vectorize=True)
        for i, j in x:
            y[i, j] = x[i, j]

    @qd.kernel
    def verify():
        for i, j in qd.ndrange(
            (boundary_offset, N - boundary_offset),
            (boundary_offset, N - boundary_offset),
        ):
            assert y[i, j] == x[i, j]

    init()
    assign_vectorized()
    verify()


@test_utils.test(require=qd.extension.quant, debug=True)
def test_offset_load():
    qu1 = qd.types.quant.int(1, False)

    x = qd.field(dtype=qu1)
    y = qd.field(dtype=qu1)
    z = qd.field(dtype=qu1)

    N = 4096
    n_blocks = 4
    bits = 32
    boundary_offset = 1024
    assert boundary_offset >= N // n_blocks

    block = qd.root.pointer(qd.ij, (n_blocks, n_blocks))
    block.dense(qd.ij, (N // n_blocks, N // (bits * n_blocks))).quant_array(qd.j, bits, max_num_bits=bits).place(x)
    block.dense(qd.ij, (N // n_blocks, N // (bits * n_blocks))).quant_array(qd.j, bits, max_num_bits=bits).place(y)
    block.dense(qd.ij, (N // n_blocks, N // (bits * n_blocks))).quant_array(qd.j, bits, max_num_bits=bits).place(z)

    @qd.kernel
    def init():
        for i, j in qd.ndrange(
            (boundary_offset, N - boundary_offset),
            (boundary_offset, N - boundary_offset),
        ):
            x[i, j] = qd.random(dtype=qd.i32) % 2

    @qd.kernel
    def assign_vectorized(dx: qd.template(), dy: qd.template()):
        qd.loop_config(bit_vectorize=True)
        for i, j in x:
            y[i, j] = x[i + dx, j + dy]
            z[i, j] = x[i + dx, j + dy]

    @qd.kernel
    def verify(dx: qd.template(), dy: qd.template()):
        for i, j in qd.ndrange(
            (boundary_offset, N - boundary_offset),
            (boundary_offset, N - boundary_offset),
        ):
            assert y[i, j] == x[i + dx, j + dy]

    init()
    assign_vectorized(0, 1)
    verify(0, 1)
    assign_vectorized(1, 0)
    verify(1, 0)
    assign_vectorized(0, -1)
    verify(0, -1)
    assign_vectorized(-1, 0)
    verify(-1, 0)
    assign_vectorized(1, 1)
    verify(1, 1)
    assign_vectorized(1, -1)
    verify(1, -1)
    assign_vectorized(-1, -1)
    verify(-1, -1)
    assign_vectorized(-1, 1)
    verify(-1, 1)


# FIXME:
#   this test fails after we introduced type u1. Actually before we introduced u1 to quadrants, this test has already
#   appeared to be problematic. All problems are related to this code:
#   `y[i, j] = (num_active_neighbors == 3) | ((num_active_neighbors == 2) & (x[i, j] == 1))`
#   Before we introduce new type u1, problems arise when:
#   1. Replace | and & with `or` and `and`
#   2. Wrap this expression with `1 if ... else 0
#   After we introduced new type u1, we can't pass this test with or without those modifications.
#   Some experiments had been carried out on this problem. The results are as follows.
#   +--------+-------------------------------+--------------------------------+---------------+
#   | (y, z) | Replace `|``&` with `or``and` |  Wrap expr with `1 if ... 0`   |  Do nothing   |
#   +--------+-------------------------------+--------------------------------+---------------+
#   | Before | always (0, 1)                 | often (0, 1), sometimes (1, 0) | OK            |
#   | After  | always (0, 1)                 | always(0, 1)                   | always (0, 1) |
#   +--------+-------------------------------+--------------------------------+---------------+
# @test_utils.test(require=qd.extension.quant, debug=True)
# def test_evolve():
#     qu1 = qd.types.quant.int(1, False)
#
#     x = qd.field(dtype=qu1)
#     y = qd.field(dtype=qu1)
#     z = qd.field(dtype=qu1)
#
#     N = 4096
#     n_blocks = 4
#     bits = 32
#     boundary_offset = 1024
#     assert boundary_offset >= N // n_blocks
#
#     block = qd.root.pointer(qd.ij, (n_blocks, n_blocks))
#     block.dense(qd.ij, (N // n_blocks, N // (bits * n_blocks))).quant_array(qd.j, bits, max_num_bits=bits).place(x)
#     block.dense(qd.ij, (N // n_blocks, N // (bits * n_blocks))).quant_array(qd.j, bits, max_num_bits=bits).place(y)
#     block.dense(qd.ij, (N // n_blocks, N // (bits * n_blocks))).quant_array(qd.j, bits, max_num_bits=bits).place(z)
#
#     @qd.kernel
#     def init():
#         for i, j in qd.ndrange(
#             (boundary_offset, N - boundary_offset),
#             (boundary_offset, N - boundary_offset),
#         ):
#             x[i, j] = qd.random(dtype=qd.i32) % 2
#
#     @qd.kernel
#     def evolve_vectorized(x: qd.template(), y: qd.template()):
#         qd.loop_config(bit_vectorize=True)
#         for i, j in x:
#             num_active_neighbors = 0
#             num_active_neighbors += qd.cast(x[i - 1, j - 1], qd.u32)
#             num_active_neighbors += qd.cast(x[i - 1, j], qd.u32)
#             num_active_neighbors += qd.cast(x[i - 1, j + 1], qd.u32)
#             num_active_neighbors += qd.cast(x[i, j - 1], qd.u32)
#             num_active_neighbors += qd.cast(x[i, j + 1], qd.u32)
#             num_active_neighbors += qd.cast(x[i + 1, j - 1], qd.u32)
#             num_active_neighbors += qd.cast(x[i + 1, j], qd.u32)
#             num_active_neighbors += qd.cast(x[i + 1, j + 1], qd.u32)
#             y[i, j] = (num_active_neighbors == 3) | ((num_active_neighbors == 2) & (x[i, j] == 1))
#
#     @qd.kernel
#     def evolve_naive(x: qd.template(), y: qd.template()):
#         for i, j in qd.ndrange(
#             (boundary_offset, N - boundary_offset),
#             (boundary_offset, N - boundary_offset),
#         ):
#             num_active_neighbors = 0
#             num_active_neighbors += qd.cast(x[i - 1, j - 1], qd.u32)
#             num_active_neighbors += qd.cast(x[i - 1, j], qd.u32)
#             num_active_neighbors += qd.cast(x[i - 1, j + 1], qd.u32)
#             num_active_neighbors += qd.cast(x[i, j - 1], qd.u32)
#             num_active_neighbors += qd.cast(x[i, j + 1], qd.u32)
#             num_active_neighbors += qd.cast(x[i + 1, j - 1], qd.u32)
#             num_active_neighbors += qd.cast(x[i + 1, j], qd.u32)
#             num_active_neighbors += qd.cast(x[i + 1, j + 1], qd.u32)
#             y[i, j] = (num_active_neighbors == 3) or (num_active_neighbors == 2 and x[i, j] == 1)
#
#     @qd.kernel
#     def verify():
#         for i, j in qd.ndrange(
#             (boundary_offset, N - boundary_offset),
#             (boundary_offset, N - boundary_offset),
#         ):
#             assert y[i, j] == z[i, j]
#
#     init()
#     evolve_naive(x, z)
#     evolve_vectorized(x, y)
#     verify()
