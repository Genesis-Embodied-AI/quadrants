import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("precision", [32, 64])
@test_utils.test(arch=[qd.cpu, qd.cuda], require=qd.extension.data64)
def test_inlined_inertia_without_reassociation(precision):
    # The diagnostic controls are absent from cache keys: isolate each configuration in a fresh process.
    env = dict(os.environ, QD_EXPERIMENT_NO_REASSOC="1", QD_EXPERIMENT_NO_CONTRACT="0")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH")]))
    arch = qd.lang.impl.current_cfg().arch.name
    proc = subprocess.run(
        [sys.executable, __file__, arch, str(precision)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def check_inlined_inertia(arch_name, precision):
    # Preserve the issue's seeded input and branch-dependent rotation matrix: the surrounding code changes how
    # reassociation orders its sums. Keep contraction enabled to catch an unnecessary dependence on disabling FMA.
    N = 20000
    FP = qd.f32 if precision == 32 else qd.f64
    qd.init(
        arch=getattr(qd, arch_name),
        default_fp=FP,
        fast_math=True,
        offline_cache=False,
        src_ll_cache=False,
        enable_fallback=False,
        cpu_max_num_threads=4,
    )

    @qd.func
    def quat_mul(u, v):
        return qd.Vector(
            [
                u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
                u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
                u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
                u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
            ]
        )

    @qd.func
    def rotate(v, q):
        w, x, y, z = q
        return qd.Vector(
            [
                v.x * (x * x + w * w - y * y - z * z)
                + v.y * (2.0 * x * y - 2.0 * w * z)
                + v.z * (2.0 * x * z + 2.0 * w * y),
                v.x * (2.0 * w * z + 2.0 * x * y)
                + v.y * (w * w - x * x + y * y - z * z)
                + v.z * (-2.0 * w * x + 2.0 * y * z),
                v.x * (-2.0 * w * y + 2.0 * x * z)
                + v.y * (2.0 * w * x + 2.0 * y * z)
                + v.z * (w * w - x * x - y * y + z * z),
            ]
        ) / (w * w + x * x + y * y + z * z)

    @qd.func
    def quat_to_R(q, eps):
        R = qd.Matrix.identity(FP, 3)
        d = q.norm_sqr()
        if d > eps:
            s = 2.0 / d
            w, x, y, z = q
            xs, ys, zs = x * s, y * s, z * s
            wx, wy, wz = w * xs, w * ys, w * zs
            xx, xy, xz = x * xs, x * ys, x * zs
            yy, yz, zz = y * ys, y * zs, z * zs
            R = qd.Matrix(
                [
                    [1.0 - (yy + zz), xy - wz, xz + wy],
                    [xy + wz, 1.0 - (xx + zz), yz - wx],
                    [xz - wy, yz + wx, 1.0 - (xx + yy)],
                ]
            )
        return R

    # Pose of a body, and pose and inertia of its center of mass in the body frame
    body_pos = qd.Vector.field(3, FP, N)
    body_quat = qd.Vector.field(4, FP, N)
    com_pos = qd.Vector.field(3, FP, N)
    com_quat = qd.Vector.field(4, FP, N)
    inertia = qd.Matrix.field(3, 3, FP, N)
    mass = qd.field(FP, N)
    origin = qd.Vector.field(3, FP, N)
    eps = qd.field(FP, ())
    # World pose of the center of mass, and inertia about the origin
    world_pos = qd.Vector.field(3, FP, N)
    world_quat = qd.Vector.field(4, FP, N)
    out = qd.Matrix.field(3, 3, FP, N)

    @qd.func
    def place_com(i):
        world_pos[i] = body_pos[i] + rotate(com_pos[i], body_quat[i])
        world_quat[i] = quat_mul(body_quat[i], com_quat[i]).normalized()

    @qd.func
    def inertia_about_origin(i):
        p = world_pos[i] - origin[i]
        hh = qd.Matrix(
            [
                [p.y * p.y + p.z * p.z, -p.x * p.y, -p.x * p.z],
                [-p.x * p.y, p.x * p.x + p.z * p.z, -p.y * p.z],
                [-p.x * p.z, -p.y * p.z, p.x * p.x + p.y * p.y],
            ]
        )
        R = quat_to_R(world_quat[i], eps[None])
        out[i] = R @ inertia[i] @ R.transpose() + hh * mass[i]

    @qd.kernel
    def two_loops():
        for i in range(N):
            place_com(i)
        for i in range(N):
            inertia_about_origin(i)

    @qd.kernel
    def one_loop():
        for i in range(N):
            place_com(i)
            inertia_about_origin(i)

    rng = np.random.default_rng(0)
    for f in (body_quat, com_quat):
        q = rng.normal(size=(N, 4))
        f.from_numpy(q / np.linalg.norm(q, axis=1, keepdims=True))
    body_pos.from_numpy(rng.normal(size=(N, 3)))
    com_pos.from_numpy(0.1 * rng.normal(size=(N, 3)))
    origin.from_numpy(rng.normal(size=(N, 3)))
    a = 0.1 * rng.normal(size=(N, 3, 3))
    inertia.from_numpy(a @ a.transpose(0, 2, 1))
    mass.from_numpy(rng.uniform(0.5, 3.0, size=N))
    eps[None] = 1e-12

    two_loops()
    pos_ref, quat_ref, out_ref = world_pos.to_numpy(), world_quat.to_numpy(), out.to_numpy()
    one_loop()

    for field, reference in ((world_pos, pos_ref), (world_quat, quat_ref), (out, out_ref)):
        actual = field.to_numpy()
        assert np.isfinite(reference).all() and np.isfinite(actual).all()
        assert actual.tobytes() == reference.tobytes(), "Identical inputs must produce bitwise identical inertia"


if __name__ == "__main__":
    check_inlined_inertia(sys.argv[1], int(sys.argv[2]))
