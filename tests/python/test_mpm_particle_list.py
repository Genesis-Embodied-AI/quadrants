import random

import pytest

import quadrants as qd

from tests import test_utils


@qd.data_oriented
class MPMSolver:
    def __init__(self, res):
        dim = len(res)
        self.dx = 1 / res[0]
        self.inv_dx = 1.0 / self.dx
        self.pid = qd.field(qd.i32)
        self.x = qd.Vector.field(dim, dtype=qd.f32)
        self.grid_m = qd.field(dtype=qd.f32)

        indices = qd.ij

        self.grid = qd.root.pointer(indices, 32)
        block = self.grid.pointer(indices, 16)
        voxel = block.dense(indices, 8)

        voxel.place(self.grid_m)
        block.dynamic(qd.axes(dim), 1024 * 1024, chunk_size=4096).place(self.pid)

        qd.root.dynamic(qd.i, 2**25, 2**20).place(self.x)
        self.substeps = 0

        for i in range(10000):
            self.x[i] = [random.random() * 0.5, random.random() * 0.5]

    @qd.kernel
    def build_pid(self):
        qd.loop_config(block_dim=256)
        for p in self.x:
            base = qd.floor(self.x[p] * self.inv_dx - 0.5).cast(int) + 1
            qd.append(self.pid.parent(), base, p)

    def step(self):
        for i in range(1000):
            self.substeps += 1
            self.grid.deactivate_all()
            self.build_pid()


@pytest.mark.run_in_serial
@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal], device_memory_GB=1.0)
def test_mpm_particle_list_no_leakage():
    # By default Quadrants will allocate 0.5 GB for testing.
    mpm = MPMSolver(res=(128, 128))
    mpm.step()
