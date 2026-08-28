import quadrants as qd

from tests import test_utils


@test_utils.test(arch=qd.cpu, require=qd.extension.sparse)
def test_memory_profiler_prints_sparse_tree_info():
    x = qd.field(qd.i32)
    fb = qd.FieldsBuilder()
    fb.pointer(qd.i, 4).place(x)
    fb.finalize()
    x[0] = 1

    qd.profiler.print_memory_profiler_info()
