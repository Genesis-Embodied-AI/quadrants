import quadrants as qd
from microbenchmarks._items import Container, DataSize, DataType
from microbenchmarks._metric import MetricType
from microbenchmarks._plan import BenchmarkPlan
from microbenchmarks._utils import dtype_size, scaled_repeat_times


def fill_default(arch, repeat, container, dtype, dsize, get_metric):
    @qd.kernel
    def fill_field(dst: qd.template()):
        for I in qd.grouped(dst):
            dst[I] = qd.cast(0.7, dtype)

    @qd.kernel
    def fill_array(dst: qd.types.ndarray()):
        for i in dst:
            dst[i] = qd.cast(0.7, dtype)

    repeat = scaled_repeat_times(arch, dsize, repeat)
    num_elements = dsize // dtype_size(dtype)
    x = container(dtype, num_elements)
    func = fill_field if container == qd.field else fill_array
    return get_metric(repeat, func, x)


def fill_sparse(arch, repeat, container, dtype, dsize, get_metric):
    repeat = scaled_repeat_times(arch, dsize, repeat=1)
    # basic_repeat_time = 1: sparse-specific parameter
    num_elements = dsize // dtype_size(dtype) // 8

    block = qd.root.pointer(qd.i, num_elements)
    x = qd.field(dtype)
    block.dense(qd.i, 8).place(x)

    @qd.kernel
    def active_all():
        for i in qd.ndrange(num_elements):
            qd.activate(block, [i])

    active_all()

    @qd.kernel
    def fill_const(dst: qd.template()):
        for i in x:
            dst[i] = qd.cast(0.7, dtype)

    return get_metric(repeat, fill_const, x)


class FillPlan(BenchmarkPlan):
    def __init__(self, arch: str):
        super().__init__("fill", arch, basic_repeat_times=10)
        fill_container = Container()
        fill_container.update({"sparse": None})  # None: implement by feature
        self.create_plan(fill_container, DataType(), DataSize(), MetricType())
        # use tag_list to label the customized implementation (funcs).
        self.add_func(["field"], fill_default)
        self.add_func(["ndarray"], fill_default)
        self.add_func(["sparse"], fill_sparse)
