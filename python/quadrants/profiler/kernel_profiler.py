# type: ignore

from contextlib import contextmanager

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.profiler.kernel_metrics import default_cupti_metrics


class StatisticalResult:
    """Statistical result of records.

    Profiling records with the same kernel name will be counted in a ``StatisticalResult`` instance via function ``insert_record(time)``.
    Currently, only the kernel elapsed time is counted, other statistics related to the kernel will be added in the feature.
    """

    def __init__(self, name):
        self.name = name
        self.counter = 0
        self.min_time = 0.0
        self.max_time = 0.0
        self.total_time = 0.0

    def __lt__(self, other):
        # For sorted()
        return self.total_time < other.total_time

    def insert_record(self, time):
        """Insert records with the same kernel name.

        Currently, only the kernel elapsed time is counted.
        """
        if self.counter == 0:
            self.min_time = time
            self.max_time = time
        self.counter += 1
        self.total_time += time
        self.min_time = min(self.min_time, time)
        self.max_time = max(self.max_time, time)


class KernelProfiler:
    """Kernel profiler of Quadrants.

    Kernel profiler acquires kernel profiling records from backend, counts records in Python scope,
    and prints the results to the console by :func:`~quadrants.profiler.kernel_profiler.KernelProfiler.print_info`.

    ``KernelProfiler`` now support detailed low-level performance metrics (such as memory bandwidth consumption) in its advanced mode.
    This mode is only available for the CUDA backend with CUPTI toolkit, i.e. you need ``qd.init(kernel_profiler=True, arch=qd.cuda)``.

    Note:
        For details about using CUPTI in Quadrants, please visit https://docs.taichi-lang.org/docs/profiler#advanced-mode.
    """

    def __init__(self):
        self._profiling_mode = False
        self._profiling_toolkit = "default"
        self._metric_list = [default_cupti_metrics]
        self._total_time_ms = 0.0
        self._traced_records = []
        self._statistical_results = {}

    # public methods

    def set_kernel_profiler_mode(self, mode=False):
        """Turn on or off :class:`~quadrants.profiler.kernel_profiler.KernelProfiler`."""
        if type(mode) is bool:
            self._profiling_mode = mode
        else:
            raise TypeError(f"Arg `mode` must be of type boolean. Type {type(mode)} is not supported.")

    def get_kernel_profiler_mode(self):
        """Get status of :class:`~quadrants.profiler.kernel_profiler.KernelProfiler`."""
        return self._profiling_mode

    def set_toolkit(self, toolkit_name="default"):
        if self._check_not_turned_on_with_warning_message():
            return False
        status = impl.get_runtime().prog.set_kernel_profiler_toolkit(toolkit_name)
        if status is True:
            self._profiling_toolkit = toolkit_name
        else:
            _qd_core.warn(
                f"Failed to set kernel profiler toolkit ({toolkit_name}) , keep using ({self._profiling_toolkit})."
            )
        return status

    def get_total_time(self):
        """Get elapsed time of all kernels recorded in KernelProfiler.

        Returns:
            time (float): total time in second.
        """
        if self._check_not_turned_on_with_warning_message():
            return 0.0
        self._update_records()  # kernel records
        self._count_statistics()  # _total_time_ms is counted here
        return self._total_time_ms / 1000  # ms to s

    def clear_info(self):
        """Clear all records both in front-end :class:`~quadrants.profiler.kernel_profiler.KernelProfiler` and back-end instance ``KernelProfilerBase``.

        Note:
            The values of ``self._profiling_mode`` and ``self._metric_list`` will not be cleared.
        """
        if self._check_not_turned_on_with_warning_message():
            return None
        # sync first
        impl.get_runtime().prog.sync_kernel_profiler()
        # then clear backend & frontend info
        impl.get_runtime().prog.clear_kernel_profiler()
        self._clear_frontend()

        return None

    def query_info(self, name):
        """For docstring of this function, see :func:`~quadrants.profiler.query_kernel_profiler_info`."""
        if self._check_not_turned_on_with_warning_message():
            return None
        self._update_records()  # kernel records
        self._count_statistics()  # statistics results
        # TODO : query self.StatisticalResult in python scope
        return impl.get_runtime().prog.query_kernel_profile_info(name)

    def set_metrics(self, metric_list=default_cupti_metrics):
        """For docstring of this function, see :func:`~quadrants.profiler.set_kernel_profiler_metrics`."""
        if self._check_not_turned_on_with_warning_message():
            return None
        self._metric_list = metric_list
        metric_name_list = [metric.name for metric in metric_list]
        self.clear_info()
        impl.get_runtime().prog.reinit_kernel_profiler_with_metrics(metric_name_list)

        return None

    @contextmanager
    def collect_metrics_in_context(self, metric_list=default_cupti_metrics):
        """This function is not exposed to user now.

        For usage of this function, see :func:`~quadrants.profiler.collect_kernel_profiler_metrics`.
        """
        if self._check_not_turned_on_with_warning_message():
            return None
        self.set_metrics(metric_list)
        yield self
        self.set_metrics()  # back to default metric list

        return None

    # mode of print_info
    COUNT = "count"  # print the statistical results (min,max,avg time) of Quadrants kernels.
    TRACE = "trace"  # print the records of launched Quadrants kernels with specific profiling metrics (time, memory load/store and core utilization etc.)

    def print_info(self, mode=COUNT):
        """Print the profiling results of Quadrants kernels.

        For usage of this function, see :func:`~quadrants.profiler.print_kernel_profiler_info`.

        Args:
            mode (str): the way to print profiling results.
        """
        if self._check_not_turned_on_with_warning_message():
            return None
        self._update_records()  # kernel records
        self._count_statistics()  # statistics results

        # COUNT mode (default) : print statistics of all kernel
        if mode == self.COUNT:
            self._print_statistics_info()
        # TRACE mode : print records of launched kernel
        elif mode == self.TRACE:
            self._print_kernel_info()
        else:
            raise ValueError("Arg `mode` must be of type 'str', and has the value 'count' or 'trace'.")

        return None

    # private methods
    def _check_not_turned_on_with_warning_message(self):
        if self._profiling_mode is False:
            _qd_core.warn("use 'qd.init(kernel_profiler = True)' to turn on KernelProfiler.")
            return True
        return False

    def _clear_frontend(self):
        """Clear member variables in :class:`~quadrants.profiler.kernel_profiler.KernelProfiler`.

        Note:
            The values of ``self._profiling_mode`` and ``self._metric_list`` will not be cleared.
        """
        self._total_time_ms = 0.0
        self._traced_records.clear()
        self._statistical_results.clear()

    def _update_records(self):
        """Acquires kernel records from a backend."""
        impl.get_runtime().prog.sync_kernel_profiler()
        impl.get_runtime().prog.update_kernel_profiler()
        self._clear_frontend()
        self._traced_records = impl.get_runtime().prog.get_kernel_profiler_records()

    def _count_statistics(self):
        """Counts the statistics of launched kernels during the profiling period.

        The profiling records with the same kernel name are counted as a profiling result.
        """
        for record in self._traced_records:
            if self._statistical_results.get(record.name) is None:
                self._statistical_results[record.name] = StatisticalResult(record.name)
            self._statistical_results[record.name].insert_record(record.kernel_time)
            self._total_time_ms += record.kernel_time
        self._statistical_results = {
            k: v
            for k, v in sorted(
                self._statistical_results.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    def _make_table_header(self, mode):
        header_str = f"Kernel Profiler({mode}, {self._profiling_toolkit})"
        arch_name = f" @ {_qd_core.arch_name(impl.current_cfg().arch).upper()}"
        device_name = impl.get_runtime().prog.get_kernel_profiler_device_name()
        if len(device_name) > 1:  # default device_name = ' '
            device_name = " on " + device_name
        return header_str + arch_name + device_name

    def _print_statistics_info(self):
        """Print statistics of launched kernels during the profiling period."""

        # headers
        table_header = table_header = self._make_table_header("count")
        column_header = "[      %     total   count |      min       avg       max   ] Kernel name"
        # partition line
        line_length = max(len(column_header), len(table_header))
        outer_partition_line = "=" * line_length
        inner_partition_line = "-" * line_length

        # message in one line
        string_list = []
        values_list = []
        for key in self._statistical_results:
            result = self._statistical_results[key]
            fraction = result.total_time / self._total_time_ms * 100.0
            string_list.append("[{:6.2f}% {:7.3f} s {:6d}x |{:9.3f} {:9.3f} {:9.3f} ms] {}")
            values_list.append(
                [
                    fraction,
                    result.total_time / 1000.0,
                    result.counter,
                    result.min_time,
                    result.total_time / result.counter,  # avg_time
                    result.max_time,
                    result.name,
                ]
            )

        # summary
        summary_line = "[100.00%] Total execution time: "
        summary_line += f"{self._total_time_ms/1000:7.3f} s   "
        summary_line += f"number of results: {len(self._statistical_results)}"

        # print
        print(outer_partition_line)
        print(table_header)
        print(outer_partition_line)
        print(column_header)
        print(inner_partition_line)
        result_num = len(self._statistical_results)
        for idx in range(result_num):
            print(string_list[idx].format(*values_list[idx]))
        print(inner_partition_line)
        print(summary_line)
        print(outer_partition_line)

    def _print_kernel_info(self):
        """Print a list of launched kernels during the profiling period."""
        metric_list = self._metric_list
        values_num = len(self._traced_records[0].metric_values)

        # We currently get kernel attributes through CUDA Driver API,
        # there is no corresponding implementation in other backends yet.
        # Profiler dose not print invalid kernel attributes info for now.
        kernel_attribute_state = self._traced_records[0].register_per_thread > 0

        # headers
        table_header = self._make_table_header("trace")
        column_header = "[  start.time | kernel.time |"  # default
        if kernel_attribute_state:
            column_header += "   regs  |   shared mem | grid size | block size | occupancy |"  # kernel_attributes
        for idx in range(values_num):
            column_header += metric_list[idx].header + "|"
        column_header = (column_header + "] Kernel name").replace("|]", "]")

        # partition line
        line_length = max(len(column_header), len(table_header))
        outer_partition_line = "=" * line_length
        inner_partition_line = "-" * line_length

        # message in one line: formatted_str.format(*values)
        fake_timestamp = 0.0
        string_list = []
        values_list = []
        for record in self._traced_records:
            formatted_str = "[{:9.3f} ms |{:9.3f} ms |"  # default
            values = [fake_timestamp, record.kernel_time]  # default
            if kernel_attribute_state:
                formatted_str += "    {:4d} | {:6d} bytes |    {:6d} |     {:6d} | {:2d} blocks |"
                values += [
                    record.register_per_thread,
                    record.shared_mem_per_block,
                    record.grid_size,
                    record.block_size,
                    record.active_blocks_per_multiprocessor,
                ]
            for idx in range(values_num):
                formatted_str += metric_list[idx].val_format + "|"
                values += [record.metric_values[idx] * metric_list[idx].scale]
            formatted_str = formatted_str + "] " + record.name
            string_list.append(formatted_str.replace("|]", "]"))
            values_list.append(values)
            fake_timestamp += record.kernel_time

        # print
        print(outer_partition_line)
        print(table_header)
        print(outer_partition_line)
        print(column_header)
        print(inner_partition_line)
        record_num = len(self._traced_records)
        for idx in range(record_num):
            print(string_list[idx].format(*values_list[idx]))
        print(inner_partition_line)
        print(f"Number of records:  {len(self._traced_records)}")
        print(outer_partition_line)


_qd_kernel_profiler = KernelProfiler()


def get_default_kernel_profiler():
    """We have only one :class:`~quadrants.profiler.kernelprofiler.KernelProfiler` instance(i.e. ``_qd_kernel_profiler``) now.

    For ``KernelProfiler`` using ``CuptiToolkit``, GPU devices can only work in a certain configuration.
    Profiling mode and metrics are configured by the host(CPU) via CUPTI APIs, and device(GPU) will use
    its counter registers to collect specific metrics.
    So if there are multiple instances of ``KernelProfiler``, the device will work in the latest configuration,
    the profiling configuration of other instances will be changed as a result.
    For data retention purposes, multiple instances will be considered in the future.
    """
    return _qd_kernel_profiler


def print_kernel_profiler_info(mode="count"):
    """Print the profiling results of Quadrants kernels.

    To enable this profiler, set ``kernel_profiler=True`` in ``qd.init()``.
    ``'count'`` mode: print the statistics (min,max,avg time) of launched kernels,
    ``'trace'`` mode: print the records of launched kernels with specific profiling metrics (time, memory load/store and core utilization etc.),
    and defaults to ``'count'``.

    Args:
        mode (str): the way to print profiling results.

    Example::

        >>> import quadrants as qd

        >>> qd.init(qd.cpu, kernel_profiler=True)
        >>> var = qd.field(qd.f32, shape=1)

        >>> @qd.kernel
        >>> def compute():
        >>>     var[0] = 1.0

        >>> compute()
        >>> qd.profiler.print_kernel_profiler_info()
        >>> # equivalent calls :
        >>> # qd.profiler.print_kernel_profiler_info('count')

        >>> qd.profiler.print_kernel_profiler_info('trace')

    Note:
        For advanced mode of `KernelProfiler`, please visit https://docs.taichi-lang.org/docs/profiler#advanced-mode.
    """
    get_default_kernel_profiler().print_info(mode)


def query_kernel_profiler_info(name):
    """Query kernel elapsed time(min,avg,max) on devices using the kernel name.

    To enable this profiler, set `kernel_profiler=True` in `qd.init`.

    Args:
        name (str): kernel name.

    Returns:
        KernelProfilerQueryResult (class): with member variables(counter, min, max, avg)

    Example::

        >>> import quadrants as qd

        >>> qd.init(qd.cpu, kernel_profiler=True)
        >>> n = 1024*1024
        >>> var = qd.field(qd.f32, shape=n)

        >>> @qd.kernel
        >>> def fill():
        >>>     for i in range(n):
        >>>         var[i] = 0.1

        >>> fill()
        >>> qd.profiler.clear_kernel_profiler_info() #[1]
        >>> for i in range(100):
        >>>     fill()
        >>> query_result = qd.profiler.query_kernel_profiler_info(fill.__name__) #[2]
        >>> print("kernel executed times =",query_result.counter)
        >>> print("kernel elapsed time(min_in_ms) =",query_result.min)
        >>> print("kernel elapsed time(max_in_ms) =",query_result.max)
        >>> print("kernel elapsed time(avg_in_ms) =",query_result.avg)

    Note:
        [1] To get the correct result, query_kernel_profiler_info() must be used in conjunction with
        clear_kernel_profiler_info().
    """
    return get_default_kernel_profiler().query_info(name)


def clear_kernel_profiler_info():
    """Clear all KernelProfiler records."""
    get_default_kernel_profiler().clear_info()


def get_kernel_profiler_total_time():
    """Get elapsed time of all kernels recorded in KernelProfiler.

    Returns:
        time (float): total time in second.
    """
    return get_default_kernel_profiler().get_total_time()


def set_kernel_profiler_toolkit(toolkit_name="default"):
    """Set the toolkit used by KernelProfiler.

    Currently, we only support toolkits: ``'default'`` and ``'cupti'``.

    Args:
        toolkit_name (str): string of toolkit name.

    Returns:
        status (bool): whether the setting is successful or not.

    Example::

        >>> import quadrants as qd

        >>> qd.init(arch=qd.cuda, kernel_profiler=True)
        >>> x = qd.field(qd.f32, shape=1024*1024)

        >>> @qd.kernel
        >>> def fill():
        >>>     for i in x:
        >>>         x[i] = i

        >>> qd.profiler.set_kernel_profiler_toolkit('cupti')
        >>> for i in range(100):
        >>>     fill()
        >>> qd.profiler.print_kernel_profiler_info()

        >>> qd.profiler.set_kernel_profiler_toolkit('default')
        >>> for i in range(100):
        >>>     fill()
        >>> qd.profiler.print_kernel_profiler_info()
    """
    return get_default_kernel_profiler().set_toolkit(toolkit_name)


def set_kernel_profiler_metrics(metric_list=default_cupti_metrics):
    """Set metrics that will be collected by the CUPTI toolkit.

    Args:
        metric_list (list): a list of :class:`~quadrants.profiler.CuptiMetric()` instances, default value: :data:`~quadrants.profiler.kernel_metrics.default_cupti_metrics`.

    Example::

        >>> import quadrants as qd

        >>> qd.init(kernel_profiler=True, arch=qd.cuda)
        >>> qd.profiler.set_kernel_profiler_toolkit('cupti')
        >>> num_elements = 128*1024*1024

        >>> x = qd.field(qd.f32, shape=num_elements)
        >>> y = qd.field(qd.f32, shape=())
        >>> y[None] = 0

        >>> @qd.kernel
        >>> def reduction():
        >>>     for i in x:
        >>>         y[None] += x[i]

        >>> # In the case of not parameter, Quadrants will print its pre-defined metrics list
        >>> qd.profiler.get_predefined_cupti_metrics()
        >>> # get Quadrants pre-defined metrics
        >>> profiling_metrics = qd.profiler.get_predefined_cupti_metrics('shared_access')

        >>> global_op_atom = qd.profiler.CuptiMetric(
        >>>     name='l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum',
        >>>     header=' global.atom ',
        >>>     format='    {:8.0f} ')
        >>> # add user defined metrics
        >>> profiling_metrics += [global_op_atom]

        >>> # metrics setting will be retained until the next configuration
        >>> qd.profiler.set_kernel_profiler_metrics(profiling_metrics)
        >>> for i in range(16):
        >>>     reduction()
        >>> qd.profiler.print_kernel_profiler_info('trace')

    Note:
        Metrics setting will be retained until the next configuration.
    """
    get_default_kernel_profiler().set_metrics(metric_list)


@contextmanager
def collect_kernel_profiler_metrics(metric_list=default_cupti_metrics):
    """Set temporary metrics that will be collected by the CUPTI toolkit within this context.

    Args:
        metric_list (list): a list of :class:`~quadrants.profiler.CuptiMetric()` instances, default value: :data:`~quadrants.profiler.kernel_metrics.default_cupti_metrics`.

    Example::

        >>> import quadrants as qd

        >>> qd.init(kernel_profiler=True, arch=qd.cuda)
        >>> qd.profiler.set_kernel_profiler_toolkit('cupti')
        >>> num_elements = 128*1024*1024

        >>> x = qd.field(qd.f32, shape=num_elements)
        >>> y = qd.field(qd.f32, shape=())
        >>> y[None] = 0

        >>> @qd.kernel
        >>> def reduction():
        >>>     for i in x:
        >>>         y[None] += x[i]

        >>> # In the case of not parameter, Quadrants will print its pre-defined metrics list
        >>> qd.profiler.get_predefined_cupti_metrics()
        >>> # get Quadrants pre-defined metrics
        >>> profiling_metrics = qd.profiler.get_predefined_cupti_metrics('device_utilization')

        >>> global_op_atom = qd.profiler.CuptiMetric(
        >>>     name='l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum',
        >>>     header=' global.atom ',
        >>>     format='    {:8.0f} ')
        >>> # add user defined metrics
        >>> profiling_metrics += [global_op_atom]

        >>> # metrics setting is temporary, and will be clear when exit from this context.
        >>> with qd.profiler.collect_kernel_profiler_metrics(profiling_metrics):
        >>>     for i in range(16):
        >>>         reduction()
        >>>     qd.profiler.print_kernel_profiler_info('trace')

    Note:
        The configuration of the ``metric_list`` will be clear when exit from this context.
    """
    get_default_kernel_profiler().set_metrics(metric_list)
    yield get_default_kernel_profiler()
    get_default_kernel_profiler().set_metrics()


__all__ = [
    "clear_kernel_profiler_info",
    "collect_kernel_profiler_metrics",
    "get_kernel_profiler_total_time",
    "print_kernel_profiler_info",
    "query_kernel_profiler_info",
    "set_kernel_profiler_metrics",
    "set_kernel_profiler_toolkit",
]
