import numpy as np

import quadrants as qd
from quadrants.lang._field_initializer import _FillValue


def check_fill_values(matrix: qd.Matrix):
    values: tuple[_FillValue, ...] = (
        1,
        1.5,
        np.int32(2),
        np.float32(2.5),
        [1, 2],
        (1, 2),
        [[1, 2], [3, 4]],
        matrix,
    )
    qd.Matrix.field(2, 2, qd.f32, shape=4, fill=None)
    for value in values:
        qd.Matrix.field(2, 2, qd.f32, shape=4, fill=value)
