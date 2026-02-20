import sys

import quadrants as qd

sys.path.append("../tests/python/")

from bls_test_template import bls_particle_grid

qd.init(arch=qd.cuda, kernel_profiler=True)
bls_particle_grid(
    N=512,
    ppc=10,
    block_size=16,
    scatter=True,
    benchmark=10,
    pointer_level=2,
    use_offset=True,
)

qd.print_kernel_profile_info()
