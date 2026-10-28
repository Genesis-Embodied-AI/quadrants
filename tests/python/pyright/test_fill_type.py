import numpy as np

import quadrants as qd
from quadrants.lang._field_initializer import _FillValue


def check_fill_values():
    values: tuple[_FillValue, ...] = (
        None,
        1,
        1.5,
        np.int32(2),
        np.float32(2.5),
        [1, 2],
        (1, 2),
        [[1, 2], [3, 4]],
        qd.Matrix([[1, 2], [3, 4]]),
    )
    for value in values:
        qd.Matrix.field(2, 2, qd.f32, shape=4, fill=value)
