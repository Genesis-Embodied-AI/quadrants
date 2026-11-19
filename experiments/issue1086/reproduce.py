import argparse
import json
import os
import platform

import numpy as np

import quadrants as qd

parser = argparse.ArgumentParser()
parser.add_argument("--arch", choices=("cpu", "cuda"), required=True)
parser.add_argument("--precision", choices=("32", "64"), required=True)
parser.add_argument("--mode", choices=("fast", "no_reassoc", "no_reassoc_no_contract", "strict"), required=True)
args = parser.parse_args()
os.environ["QD_EXPERIMENT_NO_REASSOC"] = str(int(args.mode in ("no_reassoc", "no_reassoc_no_contract")))
os.environ["QD_EXPERIMENT_NO_CONTRACT"] = str(int(args.mode == "no_reassoc_no_contract"))
N = 20000
FP = qd.f32 if args.precision == "32" else qd.f64


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


for arch in (qd.cpu if args.arch == "cpu" else qd.cuda,):
    for fast_math in (args.mode != "strict",):
        qd.init(
            arch=arch,
            default_fp=FP,
            fast_math=fast_math,
            offline_cache=False,
            src_ll_cache=False,
            cpu_max_num_threads=16,
            enable_fallback=False,
        )
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
        # The inputs of 'inertia_about_origin' are bitwise identical in both kernels
        pos_equal = world_pos.to_numpy().tobytes() == pos_ref.tobytes()
        quat_equal = world_quat.to_numpy().tobytes() == quat_ref.tobytes()
        n_diff = (out.to_numpy() != out_ref).any(axis=(1, 2)).sum()
        max_diff = np.abs(out.to_numpy() - out_ref).max()
        arch_name = qd.lang.impl.current_cfg().arch.name
        print(f"{arch_name:5s} fast_math={fast_math!s:5s}: {n_diff:5d} of {N} results differ (max {max_diff:.1e})")
        result = dict(
            arch=arch_name,
            precision=args.precision,
            mode=args.mode,
            pos_equal=pos_equal,
            quat_equal=quat_equal,
            differing_rows=int(n_diff),
            max_abs=float(max_diff),
            n=N,
            hostname=platform.node(),
            version=qd.__version__,
        )
        print("RESULT " + json.dumps(result), flush=True)
        qd.reset()
