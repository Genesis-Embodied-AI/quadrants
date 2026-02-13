import quadrants as ti
from quadrants.lang._template_mapper import TemplateMapper
from quadrants.lang.kernel_arguments import ArgMetadata

from tests import test_utils


@test_utils.test()
def test_callable_template_mapper():
    x = ti.field(ti.i32)
    y = ti.field(ti.f32)

    ti.root.place(x, y)

    mapper = TemplateMapper(
        (
            ArgMetadata(ti.template(), ti.template()),
            ArgMetadata(ti.template(), ti.template()),
            ArgMetadata(ti.template(), ti.template()),
        ),
        template_slot_locations=(0, 1, 2),
    )
    assert mapper.lookup(False, (0, 0, 0))[0] == 0
    assert mapper.lookup(False, (0, 1, 0))[0] == 1
    assert mapper.lookup(False, (0, 0, 0))[0] == 0
    assert mapper.lookup(False, (0, 0, 1))[0] == 2
    assert mapper.lookup(False, (0, 1, 0))[0] == 1

    mapper = TemplateMapper(
        (
            ArgMetadata(ti.i32, ti.i32),
            ArgMetadata(ti.i32, ti.i32),
            ArgMetadata(ti.i32, ti.i32),
        ),
        (),
    )
    assert mapper.lookup(False, (0, 0, 0))[0] == 0
    assert mapper.lookup(False, (0, 1, 0))[0] == 0
    assert mapper.lookup(False, (0, 0, 0))[0] == 0
    assert mapper.lookup(False, (0, 0, 1))[0] == 0
    assert mapper.lookup(False, (0, 1, 0))[0] == 0

    mapper = TemplateMapper(
        (
            ArgMetadata(ti.i32, ti.i32),
            ArgMetadata(ti.template(), ti.template()),
            ArgMetadata(ti.i32, ti.i32),
        ),
        (1,),
    )
    assert mapper.lookup(False, (0, x, 0))[0] == 0
    assert mapper.lookup(False, (0, y, 0))[0] == 1
    assert mapper.lookup(False, (0, x, 1))[0] == 0


@test_utils.test()
def test_callable_template_mapper_numpy():
    x = ti.field(ti.i32)
    y = ti.field(ti.f32)

    ti.root.place(x, y)

    annotations = (
        ArgMetadata(ti.template(), ti.template()),
        ArgMetadata(ti.template(), ti.template()),
        ArgMetadata(ti.types.ndarray(), ti.types.ndarray()),
    )

    import numpy as np

    mapper = TemplateMapper(annotations, (0, 1, 2))
    assert mapper.lookup(False, (0, 0, np.ones(shape=(1, 2, 3), dtype=np.float32)))[0] == 0
    assert mapper.lookup(False, (0, 0, np.ones(shape=(1, 2, 4), dtype=np.float32)))[0] == 0
    assert mapper.lookup(False, (0, 0, np.ones(shape=(1, 2, 1), dtype=np.int32)))[0] == 1
