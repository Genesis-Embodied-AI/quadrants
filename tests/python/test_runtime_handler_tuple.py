import dataclasses
from collections.abc import Callable
from typing import Protocol

import numpy as np

import quadrants as qd

from tests.helpers import test_utils


@test_utils.test()
def test_runtime_handler_tuple_example():
    @dataclasses.dataclass(frozen=True)
    class GridState:
        vel: qd.Tensor

    @dataclasses.dataclass(frozen=True)
    class CouplingData:
        vel_delta: qd.Tensor

    @dataclasses.dataclass(frozen=True)
    class CPICData:
        is_cell_separated: qd.Tensor
        cells_link_idx: qd.Tensor

    @dataclasses.dataclass(frozen=True)
    class RigidState:
        links_vel: qd.Tensor

    class Handler(Protocol):
        def __call__(self, i_c: int, vel: float) -> float: ...

    @qd.data_oriented
    class Bound[*Ts]:
        def __init__(self, func: Callable[[int, float, *Ts], float], *data: *Ts) -> None:
            self.func = func
            self.data = data

        @qd.func
        def __call__(self, i_c: int, vel: float) -> float:
            return self.func(i_c, vel, *self.data)

    @qd.func
    def func_add_coupling_vel(i_c: int, vel: float, coupling_data: CouplingData) -> float:
        return vel + coupling_data.vel_delta[i_c]

    @qd.func
    def func_replace_cpic_vel(i_c: int, vel: float, cpic_data: CPICData, rigid_state: RigidState) -> float:
        i_l = cpic_data.cells_link_idx[i_c]
        return rigid_state.links_vel[i_l] if cpic_data.is_cell_separated[i_c] else vel

    @qd.kernel
    def kernel_g2p(grid_state: GridState, vel_particles: qd.Tensor, handlers: tuple[Handler, ...]) -> None:
        for i_c in range(vel_particles.shape[0]):
            vel = grid_state.vel[i_c]
            for handler in handlers:
                vel = handler(i_c, vel)
            vel_particles[i_c] = vel

    def tensor(values, dtype):
        result = qd.tensor(dtype, shape=len(values), backend=qd.Backend.NDARRAY)
        result.from_numpy(np.asarray(values, dtype=np.float32 if dtype == qd.f32 else np.int32))
        return result

    grid_state = GridState(tensor([1.0] * 4, qd.f32))
    coupling_data = CouplingData(tensor([0.5] * 4, qd.f32))
    cpic_data = CPICData(tensor([False, True, False, True], qd.i32), tensor([0, 0, 1, 1], qd.i32))
    rigid_state = RigidState(tensor([-2.0, -3.0], qd.f32))
    vel_particles = tensor([0.0] * 4, qd.f32)
    handlers: tuple[Handler, ...] = (
        Bound(func_add_coupling_vel, coupling_data),
        Bound(func_replace_cpic_vel, cpic_data, rigid_state),
    )
    kernel_g2p(grid_state, vel_particles, handlers)
    np.testing.assert_array_equal(vel_particles.to_numpy(), [1.5, -2.0, 1.5, -3.0])
