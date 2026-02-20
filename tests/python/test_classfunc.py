import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test(arch=get_host_arch_list())
def test_classfunc():
    @qd.data_oriented
    class Foo:
        def __init__(self):
            self.val = qd.Matrix.field(n=3, m=3, dtype=qd.f32, shape=3)

        @qd.func
        def add_mat(self, a, b):
            return a + b

        @qd.kernel
        def fill(self):
            self.val[0] = self.add_mat(self.val[1], self.val[2])

    foo = Foo()
    foo.fill()


@test_utils.test(arch=get_host_arch_list())
def test_class_with_field():
    @qd.data_oriented
    class B(object):
        def __init__(self):
            self.x = qd.field(int)
            fb = qd.FieldsBuilder()
            fb.dense(qd.i, 1).place(self.x)
            self.snode_tree = fb.finalize()

        def clear(self):
            self.snode_tree.destroy()

    @qd.data_oriented
    class A(object):
        def __init__(self):
            self.n = 12345

        def init(self):
            self.b = B()
            self.x = qd.field(int)
            fb = qd.FieldsBuilder()
            fb.dense(qd.i, self.n).place(self.x)
            self.snode_tree = fb.finalize()

        def clear(self):
            self.snode_tree.destroy()
            self.b.clear()
            del self.b

        @qd.kernel
        def k(self, m: int):
            for i in range(self.n):
                self.x[i] = m * i
                self.b.x[0] += m

        def start(self):
            self.init()
            self.k(1)
            assert self.x[34] == 34
            assert self.b.x[0] == 12345
            self.clear()
            del self.x

            self.init()
            self.k(2)
            assert self.x[34] == 68
            assert self.b.x[0] == 24690
            self.clear()
            del self.x

    a = A()
    a.start()
