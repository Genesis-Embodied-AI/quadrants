import quadrants as qd
from microbenchmarks._items import Container, DataSize, DataType
from microbenchmarks._metric import MetricType
from microbenchmarks._plan import BenchmarkPlan
from microbenchmarks._utils import dtype_size, fill_random, scaled_repeat_times


def saxpy_default(arch, repeat, container, dtype, dsize, get_metric):
    repeat = scaled_repeat_times(arch, dsize, repeat)
    num_elements = dsize // dtype_size(dtype) // 3  # z=x+y

    x = container(dtype, num_elements)
    y = container(dtype, num_elements)
    z = container(dtype, num_elements)

    @qd.kernel
    def saxpy_field(z: qd.template(), x: qd.template(), y: qd.template()):
        for i in z:
            z[i] = 17 * x[i] + y[i]

    @qd.kernel
    def saxpy_array(z: qd.types.ndarray(), x: qd.types.ndarray(), y: qd.types.ndarray()):
        for i in z:
            z[i] = 17 * x[i] + y[i]

    fill_random(x, dtype, container)
    fill_random(y, dtype, container)
    func = saxpy_field if container == qd.field else saxpy_array
    return get_metric(repeat, func, z, x, y)


class SaxpyPlan(BenchmarkPlan):
    def __init__(self, arch: str):
        super().__init__("saxpy", arch, basic_repeat_times=10)
        self.create_plan(Container(), DataType(), DataSize(), MetricType())
        self.add_func(["field"], saxpy_default)
        self.add_func(["ndarray"], saxpy_default)
