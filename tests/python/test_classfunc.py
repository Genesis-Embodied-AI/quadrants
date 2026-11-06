import importlib
import textwrap
import zipfile

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


_DATA_ORIENTED_MODULE_SRC = textwrap.dedent(
    """
    import quadrants as qd


    @qd.data_oriented
    class Foo:
        def __init__(self):
            self.val = qd.field(qd.i32, shape=())

        @qd.kernel
        def fill(self):
            self.val[None] = VALUE
    """
)


@test_utils.test(arch=qd.cpu)
def test_classfunc_after_module_reload(tmp_path, monkeypatch):
    module_path = tmp_path / "reloaded_data_oriented.py"
    module_path.write_text(_DATA_ORIENTED_MODULE_SRC.replace("VALUE", "1"))
    monkeypatch.syspath_prepend(tmp_path)
    module = importlib.import_module("reloaded_data_oriented")
    foo = module.Foo()
    foo.fill()
    assert foo.val[None] == 1

    # Shift every line of the class so that the source lines cached from the first import point elsewhere.
    module_path.write_text("# padding\n" * 7 + _DATA_ORIENTED_MODULE_SRC.replace("VALUE", "2"))
    module = importlib.reload(module)
    foo = module.Foo()
    foo.fill()
    assert foo.val[None] == 2


@test_utils.test(arch=qd.cpu)
def test_classfunc_in_zipimported_module(tmp_path, monkeypatch):
    archive_path = tmp_path / "kernels.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("zipped_data_oriented.py", _DATA_ORIENTED_MODULE_SRC.replace("VALUE", "3"))
    monkeypatch.syspath_prepend(archive_path)
    module = importlib.import_module("zipped_data_oriented")
    foo = module.Foo()
    foo.fill()
    assert foo.val[None] == 3
