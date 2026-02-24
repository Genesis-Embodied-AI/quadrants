import quadrants as qd
from microbenchmarks._items import BenchmarkItem
from microbenchmarks._utils import End2EndTimer, get_qd_arch


def end2end_executor(repeat, func, *args):
    # compile & warmup
    for i in range(repeat):
        func(*args)

    timer = End2EndTimer()
    timer.tick()
    for i in range(repeat):
        func(*args)
    time_in_s = timer.tock()
    return time_in_s * 1000 / repeat  # ms


def kernel_executor(repeat, func, *args):
    # compile & warmup
    for i in range(repeat):
        func(*args)
    qd.profiler.clear_kernel_profiler_info()
    for i in range(repeat):
        func(*args)
    return qd.profiler.get_kernel_profiler_total_time() * 1000 / repeat  # ms


class MetricType(BenchmarkItem):
    name = "get_metric"

    def __init__(self):
        self._items = {
            "kernel_elapsed_time_ms": kernel_executor,
            "end2end_time_ms": end2end_executor,
        }

    @staticmethod
    def init_quadrants(arch: str, tag_list: list):
        if set(["kernel_elapsed_time_ms"]).issubset(tag_list):
            qd.init(kernel_profiler=True, arch=get_qd_arch(arch))
        elif set(["end2end_time_ms"]).issubset(tag_list):
            qd.init(kernel_profiler=False, arch=get_qd_arch(arch))
        else:
            return False
        return True
