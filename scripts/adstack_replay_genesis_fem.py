"""Exercise explicit FEM backward with replay counting in a Genesis environment.

Run with Genesis installed alongside this Quadrants checkout. Use --legacy for the existing sizing baseline.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    os.environ["QD_AD_STACK_REPLAY_COUNT"] = "0" if args.legacy else "1"

    import genesis as gs
    import numpy as np

    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    dt, steps = 1e-4, 3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, requires_grad=True, gravity=(0, 0, 0)),
        fem_options=gs.options.FEMOptions(use_implicit_solver=False),
        show_viewer=False,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5)),
        material=gs.materials.FEM.Elastic(E=1e4, nu=0.3, rho=1000),
    )
    scene.build()
    velocity = gs.tensor([0.1, 0.0, 0.0], requires_grad=True)
    entity.set_velocity(velocity)
    for _ in range(steps):
        scene.step()
    state = scene.get_state().solvers_state[scene.solvers.index(scene.fem_solver)]
    loss = state.pos[..., 0].mean()
    loss.backward()
    gradient = velocity.grad.detach().cpu().numpy()
    # Uniform translation with no external forces advances the mean x coordinate by steps * dt * velocity.x.
    np.testing.assert_allclose(gradient, [steps * dt, 0, 0], rtol=1e-5, atol=1e-9)
    print({"legacy": args.legacy, "loss": loss.item(), "velocity_gradient": gradient.tolist()})
    gs.destroy()


if __name__ == "__main__":
    main()
