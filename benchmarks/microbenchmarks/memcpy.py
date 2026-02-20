import quadrants as qd
from microbenchmarks._items import Container, DataSize, DataType
from microbenchmarks._metric import MetricType
from microbenchmarks._plan import BenchmarkPlan
from microbenchmarks._utils import dtype_size, fill_random, scaled_repeat_times


def memcpy_default(arch, repeat, container, dtype, dsize, get_metric):
    @qd.kernel
    def memcpy_field(dst: qd.template(), src: qd.template()):
        for I in qd.grouped(dst):
            dst[I] = src[I]

    @qd.kernel
    def memcpy_array(dst: qd.types.ndarray(), src: qd.types.ndarray()):
        for I in qd.grouped(dst):
            dst[I] = src[I]

    repeat = scaled_repeat_times(arch, dsize, repeat)
    num_elements = dsize // dtype_size(dtype) // 2  # y=x

    x = container(dtype, num_elements)
    y = container(dtype, num_elements)

    func = memcpy_field if container == qd.field else memcpy_array
    fill_random(x, dtype, container)
    return get_metric(repeat, func, y, x)


class MemcpyPlan(BenchmarkPlan):
    def __init__(self, arch: str):
        super().__init__("memcpy", arch, basic_repeat_times=10)
        self.create_plan(Container(), DataType(), DataSize(), MetricType())
        self.add_func(["field"], memcpy_default)
        self.add_func(["ndarray"], memcpy_default)
