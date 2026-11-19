"""Measure reset/replay equality and steady-state stepping under the four math modes."""

import argparse
import functools
import json
import os
import platform
import statistics
import time

import numpy as np

import quadrants as qd

parser = argparse.ArgumentParser()
parser.add_argument("--arch", choices=("cpu", "cuda"), required=True)
parser.add_argument("--precision", choices=("32", "64"), required=True)
parser.add_argument("--mode", choices=("fast", "no_reassoc", "no_reassoc_no_contract", "strict"), required=True)
parser.add_argument("--n-envs", type=int, default=4)
parser.add_argument("--performance", action="store_true")
parser.add_argument("--scene", choices=("boxes", "anymal"), default="boxes")
parser.add_argument("--benchmark", action="store_true")
args = parser.parse_args()
os.environ["QD_EXPERIMENT_NO_REASSOC"] = str(int(args.mode in ("no_reassoc", "no_reassoc_no_contract")))
os.environ["QD_EXPERIMENT_NO_CONTRACT"] = str(int(args.mode == "no_reassoc_no_contract"))
os.environ["QD_OFFLINE_CACHE"] = "0"
original_init = qd.init


@functools.wraps(original_init)
def controlled_init(*positional, **kwargs):
    kwargs.update(fast_math=args.mode != "strict", offline_cache=False, src_ll_cache=False, cpu_max_num_threads=16)
    return original_init(*positional, **kwargs)


qd.init = controlled_init

import genesis as gs  # noqa: E402
from genesis.utils.misc import qd_to_numpy, tensor_to_array  # noqa: E402


def snapshot(scene):
    return {
        "qpos": tensor_to_array(scene.rigid_solver.get_qpos()).copy(),
        "velocity": tensor_to_array(scene.rigid_solver.get_dofs_velocity()).copy(),
        "links_pos": tensor_to_array(scene.rigid_solver.get_links_pos()).copy(),
        "cinr_inertial": qd_to_numpy(scene.rigid_solver.dyn_state.links.cinr_inertial).copy(),
    }


def compare(actual, reference):
    return {
        name: {
            "bitwise_equal": actual[name].tobytes() == reference[name].tobytes(),
            "different_elements": int(np.count_nonzero(actual[name] != reference[name])),
            "max_abs": float(np.abs(actual[name] - reference[name]).max(initial=0)),
            "finite": bool(np.isfinite(actual[name]).all() and np.isfinite(reference[name]).all()),
        }
        for name in reference
    }


gs.init(
    backend=gs.cpu if args.arch == "cpu" else gs.cuda,
    precision=args.precision,
    debug=False,
    seed=0,
    performance_mode=args.performance,
    use_deterministic_algorithms=True,
    logging_level="warning",
)
scene = gs.Scene(
    rigid_options=gs.options.RigidOptions(iterations=25),
    show_viewer=False,
    show_FPS=False,
)
scene.add_entity(gs.morphs.Plane())
if args.scene == "boxes":
    # Same two independent falling boxes as tests/rigid/test_api.py::test_reset.
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5)))
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(2, 0, 0.5)))
else:
    robot = scene.add_entity(gs.morphs.URDF(file="urdf/anymal_c/urdf/anymal_c.urdf", pos=(0, 0, 0.8)))
scene.build(n_envs=args.n_envs)
if args.scene == "anymal":
    robot.set_dofs_kp(1000.0, slice(6, None))
    robot.control_dofs_position(0.0, slice(6, None))
for _ in range(60):
    scene.step()
saved = scene.get_state()
at_capture = snapshot(scene)
reference = []
for _ in range(10):
    scene.step()
    scene.get_state()
    reference.append(snapshot(scene))
scene.reset(state=saved)
at_restore = snapshot(scene)
replay = []
for expected in reference:
    scene.step()
    scene.get_state()
    replay.append(compare(snapshot(scene), expected))
result = {
    "arch": args.arch,
    "precision": args.precision,
    "mode": args.mode,
    "performance": args.performance,
    "n_envs": args.n_envs,
    "scene": args.scene,
    "hostname": platform.node(),
    "restore": compare(at_restore, at_capture),
    "replay": replay,
}
if args.benchmark:
    # Fixed work and synchronization at timing boundaries; compilation is excluded.
    for _ in range(100):
        scene.step()
    samples = []
    for _ in range(5):
        qd.sync()
        start = time.perf_counter()
        for _ in range(200):
            scene.step()
        qd.sync()
        samples.append((time.perf_counter() - start) / 200)
    result["step_seconds"] = samples
    result["median_step_seconds"] = statistics.median(samples)
print("RESULT " + json.dumps(result), flush=True)
gs.destroy()
