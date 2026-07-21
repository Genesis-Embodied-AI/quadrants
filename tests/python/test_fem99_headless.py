"""Headless numerical repro for the alleged Metal native-float-atomic FEM99/FEM128 bug.

Background
----------
In Jan 2023 (Taichi #7093, PENGUINLIONG), when Metal switched to SPIR-V codegen, native float atomics were detected for
Apple7+/Mac2+ but immediately commented out with:

    FIXME: floating point atomics doesn't work and breaks the FEM99/FEM128 examples.

Those examples were interactive autodiff neo-Hookean soft-body demos (`python/taichi/examples/simulation/fem99.py`,
later removed from Quadrants). They were NEVER turned into a CI test, and the failure mode (wrong numbers? NaN? hang?
visual explosion?) was never written down. Upstream taichi still carries the identical FIXME.

The critical atomic pattern in FEM99 is the scalar energy reduction under autodiff::

    U[None] += V[i] * phi_i          # parallel over faces; becomes qd.atomic_add(f32)
    with qd.ad.Tape(loss=U): ...     # reverse scatter also uses float atomics into pos.grad

This file ports that pattern headlessly and checks for the symptoms we can assert without a GUI: finite energy /
positions, no blow-up, and gradients matching a CPU reference on a small case.

    QD_WANTED_ARCHS=metal pytest tests/python/test_fem99_headless.py -v
"""

from __future__ import annotations

import numpy as np

import quadrants as qd

from tests import test_utils


def _run_fem99(n_grid: int, n_frames: int, substeps: int, seed: int = 0):
    """Port of the removed fem99.py, headless. Returns (U_hist, pos_final)."""
    N = n_grid
    dt = 1e-4
    dx = 1.0 / N
    rho = 4e1
    NF = 2 * N**2
    NV = (N + 1) ** 2
    E, nu = 4e4, 0.2
    mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)
    ball_pos = qd.Vector([0.5, 0.0])
    ball_radius = 0.32
    gravity = qd.Vector([0.0, -40.0])
    damping = 12.5

    pos = qd.Vector.field(2, float, NV, needs_grad=True)
    vel = qd.Vector.field(2, float, NV)
    f2v = qd.Vector.field(3, int, NF)
    B = qd.Matrix.field(2, 2, float, NF)
    F = qd.Matrix.field(2, 2, float, NF, needs_grad=True)
    V = qd.field(float, NF)
    phi = qd.field(float, NF)
    U = qd.field(float, (), needs_grad=True)

    @qd.kernel
    def update_U():
        for i in range(NF):
            ia, ib, ic = f2v[i]
            a, b, c = pos[ia], pos[ib], pos[ic]
            V[i] = abs((a - c).cross(b - c))
            D_i = qd.Matrix.cols([a - c, b - c])
            F[i] = D_i @ B[i]
        for i in range(NF):
            F_i = F[i]
            log_J_i = qd.log(F_i.determinant())
            phi_i = mu / 2 * ((F_i.transpose() @ F_i).trace() - 2)
            phi_i -= mu * log_J_i
            phi_i += lam / 2 * log_J_i**2
            phi[i] = phi_i
            # THE atomic float reduction that motivated the Metal native-float-atomic disable.
            U[None] += V[i] * phi_i

    @qd.kernel
    def advance():
        for i in range(NV):
            acc = -pos.grad[i] / (rho * dx**2)
            vel[i] += dt * (acc + gravity)
            vel[i] *= qd.exp(-dt * damping)
        for i in range(NV):
            disp = pos[i] - ball_pos
            disp2 = disp.norm_sqr()
            if disp2 <= ball_radius**2:
                NoV = vel[i].dot(disp)
                if NoV < 0:
                    vel[i] -= NoV * disp / disp2
            cond = ((pos[i] < 0) & (vel[i] < 0)) | ((pos[i] > 1) & (vel[i] > 0))
            for j in qd.static(range(pos.n)):
                if cond[j]:
                    vel[i][j] = 0
            pos[i] += dt * vel[i]

    @qd.kernel
    def init_pos():
        for i, j in qd.ndrange(N + 1, N + 1):
            k = i * (N + 1) + j
            pos[k] = qd.Vector([i, j]) / N * 0.25 + qd.Vector([0.45, 0.45])
            vel[k] = qd.Vector([0.0, 0.0])
        for i in range(NF):
            ia, ib, ic = f2v[i]
            a, b, c = pos[ia], pos[ib], pos[ic]
            B_i_inv = qd.Matrix.cols([a - c, b - c])
            B[i] = B_i_inv.inverse()

    @qd.kernel
    def init_mesh():
        for i, j in qd.ndrange(N, N):
            k = (i * N + j) * 2
            a = i * (N + 1) + j
            b = a + 1
            c = a + N + 2
            d = a + N + 1
            f2v[k + 0] = [a, b, c]
            f2v[k + 1] = [c, d, a]

    init_mesh()
    init_pos()

    u_hist = []
    for _ in range(n_frames):
        for _ in range(substeps):
            with qd.ad.Tape(loss=U):
                update_U()
            advance()
        u_hist.append(float(U[None]))

    return np.array(u_hist, dtype=np.float64), pos.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.metal])
def test_fem99_headless_stays_finite():
    """Does the FEM99 autodiff+atomic-reduce pattern stay numerically alive on Metal?"""
    # fem99 used N=32; keep it for fidelity on Metal. CPU can take the same size.
    n_grid = 32
    n_frames = 5
    substeps = 30  # same as the original demo's per-frame substep count

    u_hist, pos = _run_fem99(n_grid=n_grid, n_frames=n_frames, substeps=substeps)

    assert np.isfinite(u_hist).all(), f"energy became non-finite: {u_hist}"
    assert np.isfinite(pos).all(), "positions became non-finite"
    # Soft body starts in [0.45,0.70]^2-ish; after a few frames under gravity it should stay roughly in the unit square
    # (the demo clamps at the walls). Explosion => |pos| >> 10.
    assert np.max(np.abs(pos)) < 10.0, f"positions exploded: max|pos|={np.max(np.abs(pos))}"
    # Energy should not blow up by many orders of magnitude frame-to-frame.
    assert np.max(np.abs(u_hist)) < 1e8, f"energy exploded: {u_hist}"
    print(f"FEM99_OK u_hist={u_hist.tolist()} max|pos|={float(np.max(np.abs(pos)))}")


@test_utils.test(arch=[qd.cpu, qd.metal])
def test_ad_scalar_atomic_reduce_matches_closed_form():
    """Smaller, sharper check: the exact `loss[None] += x[i]**2` pattern (test_ad_atomic) on Metal.

    If native float atomics corrupt either the forward reduction or the reverse scatter, this fails.
    """
    N = 64
    x = qd.field(dtype=qd.f32, shape=N, needs_grad=True)
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def func():
        for i in x:
            loss[None] += x[i] ** 2

    for i in range(N):
        x[i] = float(i) * 0.1

    with qd.ad.Tape(loss):
        func()

    expected = sum((i * 0.1) ** 2 for i in range(N))
    assert loss[None] == test_utils.approx(expected, rel=1e-4)
    for i in range(N):
        assert x.grad[i] == test_utils.approx(2 * i * 0.1, rel=1e-4)

    print(f"AD_REDUCE_OK loss={float(loss[None])}")
