import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_if():
    x = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x)

    @qd.kernel
    def func():
        if True:
            a = 0
        else:
            a = 1
        print(a)

    with pytest.raises(Exception):
        func()


@test_utils.test()
def test_for():
    x = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x)

    @qd.kernel
    def func():
        for i in range(10):
            a = i
        print(a)

    with pytest.raises(Exception):
        func()


@test_utils.test()
def test_while():
    x = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x)

    @qd.kernel
    def func():
        while True:
            a = 0
        print(a)

    with pytest.raises(Exception):
        func()
