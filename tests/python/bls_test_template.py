import random

import numpy as np

import quadrants as qd


def bls_test_template(dim, N, bs, stencil, block_dim=None, scatter=False, benchmark=0, dense=False):
    x, y, y2 = qd.field(qd.i32), qd.field(qd.i32), qd.field(qd.i32)

    index = qd.axes(*range(dim))
    mismatch = qd.field(qd.i32, shape=())

    if not isinstance(bs, (tuple, list)):
        bs = [bs for _ in range(dim)]

    grid_size = [N // bs[i] for i in range(dim)]

    if dense:
        create_block = lambda: qd.root.dense(index, grid_size)
    else:
        create_block = lambda: qd.root.pointer(index, grid_size)

    if scatter:
        block = create_block()

        block.dense(index, bs).place(x)
        block.dense(index, bs).place(y)
        block.dense(index, bs).place(y2)
    else:
        create_block().dense(index, bs).place(x)
        create_block().dense(index, bs).place(y)
        create_block().dense(index, bs).place(y2)

    ndrange = tuple((bs[i] * 2, N - bs[i] * 2) for i in range(dim))

    if block_dim is None:
        block_dim = 1
        for i in range(dim):
            block_dim *= bs[i]

    @qd.kernel
    def populate():
        for I in qd.grouped(qd.ndrange(*ndrange)):
            s = 0
            for i in qd.static(range(dim)):
                s += I[i] ** (i + 1)
            x[I] = s

    @qd.kernel
    def apply(use_bls: qd.template(), y: qd.template()):
        if qd.static(use_bls and not scatter):
            qd.block_local(x)
        if qd.static(use_bls and scatter):
            qd.block_local(y)

        qd.loop_config(block_dim=block_dim)
        for I in qd.grouped(x):
            if qd.static(scatter):
                for offset in qd.static(stencil):
                    y[I + qd.Vector(offset)] += x[I]
            else:
                # gather
                s = 0
                for offset in qd.static(stencil):
                    s = s + x[I + qd.Vector(offset)]
                y[I] = s

    if benchmark:
        for i in range(benchmark):
            x.snode.parent().deactivate_all()
            if not scatter:
                populate()
            y.snode.parent().deactivate_all()
            y2.snode.parent().deactivate_all()
            apply(False, y2)
            apply(True, y)
    else:
        # Simply test
        apply(False, y2)
        apply(True, y)

    @qd.kernel
    def check():
        for I in qd.grouped(y2):
            if y[I] != y2[I]:
                print("check failed", I, y[I], y2[I])
                mismatch[None] = 1

    check()

    qd.profiler.print_kernel_profiler_info()

    assert mismatch[None] == 0


def bls_particle_grid(
    N,
    ppc=8,
    block_size=16,
    scatter=True,
    benchmark=0,
    pointer_level=1,
    sort_points=True,
    use_offset=True,
):
    M = N * N * ppc

    m1 = qd.field(qd.f32)
    m2 = qd.field(qd.f32)
    m3 = qd.field(qd.f32)
    pid = qd.field(qd.i32)
    err = qd.field(qd.i32, shape=())

    max_num_particles_per_block = block_size**2 * 4096

    x = qd.Vector.field(2, dtype=qd.f32)

    s1 = qd.field(dtype=qd.f32)
    s2 = qd.field(dtype=qd.f32)
    s3 = qd.field(dtype=qd.f32)

    qd.root.dense(qd.i, M).place(x)
    qd.root.dense(qd.i, M).place(s1, s2, s3)

    if pointer_level == 1:
        block = qd.root.pointer(qd.ij, N // block_size)
    elif pointer_level == 2:
        block = qd.root.pointer(qd.ij, N // block_size // 4).pointer(qd.ij, 4)
    else:
        raise ValueError("pointer_level must be 1 or 2")

    if use_offset:
        grid_offset = (-N // 2, -N // 2)
        grid_offset_block = (-N // 2 // block_size, -N // 2 // block_size)
        world_offset = -0.5
    else:
        grid_offset = (0, 0)
        grid_offset_block = (0, 0)
        world_offset = 0

    block.dense(qd.ij, block_size).place(m1, offset=grid_offset)
    block.dense(qd.ij, block_size).place(m2, offset=grid_offset)
    block.dense(qd.ij, block_size).place(m3, offset=grid_offset)

    block.dynamic(qd.l, max_num_particles_per_block, chunk_size=block_size**2 * ppc * 4).place(
        pid, offset=grid_offset_block + (0,)
    )

    bound = 0.1

    extend = 4

    x_ = [
        (
            random.random() * (1 - 2 * bound) + bound + world_offset,
            random.random() * (1 - 2 * bound) + bound + world_offset,
        )
        for _ in range(M)
    ]
    if sort_points:
        x_.sort(key=lambda q: int(q[0] * N) // block_size * N + int(q[1] * N) // block_size)

    x.from_numpy(np.array(x_, dtype=np.float32))

    @qd.kernel
    def insert():
        qd.loop_config(block_dim=256)
        for i in x:
            # It is important to ensure insert and p2g uses the exact same way to compute the base
            # coordinates. Otherwise there might be coordinate mismatch due to float-point errors.
            base = qd.Vector(
                [
                    int(qd.floor(x[i][0] * N) - grid_offset[0]),
                    int(qd.floor(x[i][1] * N) - grid_offset[1]),
                ]
            )
            base_p = qd.rescale_index(m1, pid, base)
            qd.append(pid.parent(), base_p, i)

    scatter_weight = (N * N / M) * 0.01

    @qd.kernel
    def p2g(use_shared: qd.template(), m: qd.template()):
        qd.loop_config(block_dim=256)
        if qd.static(use_shared):
            qd.block_local(m)
        for I in qd.grouped(pid):
            p = pid[I]

            u_ = qd.floor(x[p] * N).cast(qd.i32)
            Im = qd.rescale_index(pid, m, I)
            u0 = qd.assume_in_range(u_[0], Im[0], 0, 1)
            u1 = qd.assume_in_range(u_[1], Im[1], 0, 1)

            u = qd.Vector([u0, u1])

            for offset in qd.static(qd.grouped(qd.ndrange(extend, extend))):
                m[u + offset] += scatter_weight

    @qd.kernel
    def p2g_naive():
        qd.loop_config(block_dim=256)
        for p in x:
            u = qd.floor(x[p] * N).cast(qd.i32)

            for offset in qd.static(qd.grouped(qd.ndrange(extend, extend))):
                m3[u + offset] += scatter_weight

    @qd.kernel
    def fill_m1():
        for i, j in qd.ndrange(N, N):
            m1[i, j] = qd.random()

    @qd.kernel
    def g2p(use_shared: qd.template(), s: qd.template()):
        qd.loop_config(block_dim=256)
        if qd.static(use_shared):
            qd.block_local(m1)
        for I in qd.grouped(pid):
            p = pid[I]

            u_ = qd.floor(x[p] * N).cast(qd.i32)

            Im = qd.rescale_index(pid, m1, I)
            u0 = qd.assume_in_range(u_[0], Im[0], 0, 1)
            u1 = qd.assume_in_range(u_[1], Im[1], 0, 1)

            u = qd.Vector([u0, u1])

            tot = 0.0

            for offset in qd.static(qd.grouped(qd.ndrange(extend, extend))):
                tot += m1[u + offset]

            s[p] = tot

    @qd.kernel
    def g2p_naive(s: qd.template()):
        qd.loop_config(block_dim=256)
        for p in x:
            u = qd.floor(x[p] * N).cast(qd.i32)

            tot = 0.0
            for offset in qd.static(qd.grouped(qd.ndrange(extend, extend))):
                tot += m1[u + offset]
            s[p] = tot

    insert()

    for i in range(benchmark):
        pid.parent(2).snode.deactivate_all()
        insert()

    @qd.kernel
    def check_m():
        for i in range(grid_offset[0], grid_offset[0] + N):
            for j in range(grid_offset[1], grid_offset[1] + N):
                if abs(m1[i, j] - m3[i, j]) > 1e-4:
                    err[None] = 1
                if abs(m2[i, j] - m3[i, j]) > 1e-4:
                    err[None] = 1

    @qd.kernel
    def check_s():
        for i in range(M):
            if abs(s1[i] - s2[i]) > 1e-4:
                err[None] = 1
            if abs(s1[i] - s3[i]) > 1e-4:
                err[None] = 1

    if scatter:
        for i in range(max(benchmark, 1)):
            p2g(True, m1)
            p2g(False, m2)
            p2g_naive()
        check_m()
    else:
        for i in range(max(benchmark, 1)):
            g2p(True, s1)
            g2p(False, s2)
            g2p_naive(s3)
        check_s()

    assert not err[None]
