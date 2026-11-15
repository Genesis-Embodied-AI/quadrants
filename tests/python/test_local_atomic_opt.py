import quadrants as qd

from tests.helpers import test_utils


@test_utils.test()
def test_cse():
    A = qd.field(qd.f32, shape=())

    @qd.kernel
    def func():
        a = 0
        a += 10
        a = a + 123
        A[()] = a

    func()
    assert A[()] == 133


@test_utils.test()
def test_store_forward():
    A = qd.field(qd.f32, shape=())

    @qd.kernel
    def func():
        a = 0
        a = 123
        a += 10
        A[()] = a

    func()
    assert A[()] == 133
