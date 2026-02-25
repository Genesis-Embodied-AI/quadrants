import atexit
import os
import shutil
import tempfile
import warnings
from copy import deepcopy as _deepcopy

from quadrants import _logging, _snode
from quadrants._lib import core as _qd_core
from quadrants._lib.core.quadrants_python import Extension
from quadrants._lib.utils import get_os_name
from quadrants.lang import impl
from quadrants.lang.expr import Expr
from quadrants.lang.impl import axes, get_runtime
from quadrants.profiler.kernel_profiler import get_default_kernel_profiler
from quadrants.types.primitive_types import f32, f64, i32, i64

warnings.filterwarnings("once", category=DeprecationWarning, module="quadrants")

# ----------------------
i = axes(0)
"""Axis 0. For multi-dimensional arrays it's the direction downward the rows.
For a 1d array it's the direction along this array.
"""
# ----------------------

j = axes(1)
"""Axis 1. For multi-dimensional arrays it's the direction across the columns.
"""
# ----------------------

k = axes(2)
"""Axis 2. For arrays of dimension `d` >= 3, view each cell as an array of
lower dimension d-2, it's the first axis of this cell.
"""
# ----------------------

l = axes(3)
"""Axis 3. For arrays of dimension `d` >= 4, view each cell as an array of
lower dimension d-2, it's the second axis of this cell.
"""
# ----------------------

ij = axes(0, 1)
"""Axes (0, 1).
"""
# ----------------------

ik = axes(0, 2)
"""Axes (0, 2).
"""
# ----------------------

il = axes(0, 3)
"""Axes (0, 3).
"""
# ----------------------

jk = axes(1, 2)
"""Axes (1, 2).
"""
# ----------------------

jl = axes(1, 3)
"""Axes (1, 3).
"""
# ----------------------

kl = axes(2, 3)
"""Axes (2, 3).
"""
# ----------------------

ijk = axes(0, 1, 2)
"""Axes (0, 1, 2).
"""
# ----------------------

ijl = axes(0, 1, 3)
"""Axes (0, 1, 3).
"""
# ----------------------

ikl = axes(0, 2, 3)
"""Axes (0, 2, 3).
"""
# ----------------------

jkl = axes(1, 2, 3)
"""Axes (1, 2, 3).
"""
# ----------------------

ijkl = axes(0, 1, 2, 3)
"""Axes (0, 1, 2, 3).
"""
# ----------------------

# ----------------------

x86_64 = _qd_core.x64
"""The x64 CPU backend.
"""
# ----------------------

x64 = _qd_core.x64
"""The X64 CPU backend.
"""
# ----------------------

arm64 = _qd_core.arm64
"""The ARM CPU backend.
"""
# ----------------------

cuda = _qd_core.cuda
"""The CUDA backend.
"""
# ----------------------

amdgpu = _qd_core.amdgpu
"""The AMDGPU backend.
"""
# ----------------------

metal = _qd_core.metal
"""The Apple Metal backend.
"""
# ----------------------

vulkan = _qd_core.vulkan
"""The Vulkan backend.
"""
# ----------------------

gpu = [cuda, metal, vulkan, amdgpu]
"""A list of GPU backends supported on the current system.
Currently contains 'cuda', 'metal', 'vulkan', 'amdgpu'.

When this is used, Quadrants automatically picks the matching GPU backend. If no
GPU is detected, Quadrants falls back to the CPU backend.
"""
# ----------------------

cpu = _qd_core.host_arch()
"""A list of CPU backends supported on the current system.
Currently contains 'x64', 'x86_64', 'arm64'.

When this is used, Quadrants automatically picks the matching CPU backend.
"""
# ----------------------


def timeline_clear():
    return impl.get_runtime().prog.timeline_clear()


def timeline_save(fn):
    return impl.get_runtime().prog.timeline_save(fn)


extension = _qd_core.Extension
"""An instance of Quadrants extension.

The list of currently available extensions is ['sparse', 'quant', \
    'mesh', 'quant_basic', 'data64', 'adstack', 'bls', 'assertion', \
        'extfunc'].
"""


def is_extension_supported(arch, ext):
    """Checks whether an extension is supported on an arch.

    Args:
        arch (quadrants_python.Arch): Specified arch.
        ext (quadrants_python.Extension): Specified extension.

    Returns:
        bool: Whether `ext` is supported on `arch`.
    """
    return _qd_core.is_extension_supported(arch, ext)


def reset():
    """Resets Quadrants to its initial state.
    This will destroy all the allocated fields and kernels, and restore
    the runtime to its default configuration.

    Example::

        >>> a = qd.field(qd.i32, shape=())
        >>> a[None] = 1
        >>> print("before reset: ", a)
        before rest: 1
        >>>
        >>> qd.reset()
        >>> print("after reset: ", a)
        # will raise error because a is unavailable after reset.
    """
    impl.reset()


class _EnvironmentConfigurator:
    def __init__(self, kwargs, _cfg):
        self.cfg = _cfg
        self.kwargs = kwargs
        self.keys = []

    def add(self, key, _cast=None):
        _cast = _cast or self.bool_int

        self.keys.append(key)

        # QD_OFFLINE_CACHE=   : no effect
        # QD_OFFLINE_CACHE=0  : False
        # QD_OFFLINE_CACHE=1  : True
        name = "QD_" + key.upper()
        value = os.environ.get(name, "")
        if key in self.kwargs:
            self[key] = self.kwargs[key]
            if value:
                _qd_core.warn(f'Environment variable {name}={value} overridden by qd.init argument "{key}"')
            del self.kwargs[key]  # mark as recognized
        elif value:
            self[key] = _cast(value)

    def __getitem__(self, key):
        return getattr(self.cfg, key)

    def __setitem__(self, key, value):
        setattr(self.cfg, key, value)

    @staticmethod
    def bool_int(x):
        return bool(int(x))


class _SpecialConfig:
    # like CompileConfig in C++, this is the configurations that belong to other submodules
    def __init__(self):
        self.log_level = "info"
        self.gdb_trigger = False
        self.short_circuit_operators = True
        self.print_full_traceback = False
        self.unrolling_limit = 32


def prepare_sandbox():
    """
    Returns a temporary directory, which will be automatically deleted on exit.
    It may contain the quadrants_python shared object or some misc. files.
    """
    tmp_dir = tempfile.mkdtemp(prefix="quadrants-")
    atexit.register(shutil.rmtree, tmp_dir)
    print(f"[Quadrants] preparing sandbox at {tmp_dir}")
    os.mkdir(os.path.join(tmp_dir, "runtime/"))
    return tmp_dir


def check_require_version(require_version):
    """
    Check if installed version meets the requirements.
    Allow to specify <major>.<minor>.<patch>.<hash>.
    <patch>.<hash> is optional. If not match, raise an exception.
    """
    # Extract version number part (i.e. toss any revision / hash parts).
    version_number_str = require_version
    for c_idx, c in enumerate(require_version):
        if not (c.isdigit() or c == "."):
            version_number_str = require_version[:c_idx]
            break
    # Get required version.
    try:
        version_number_tuple = tuple([int(n) for n in version_number_str.split(".")])
        major = version_number_tuple[0]
        minor = version_number_tuple[1]
        patch = 0
        if len(version_number_tuple) > 2:
            patch = version_number_tuple[2]
    except:
        raise Exception(
            "The require_version should be formatted following PEP 440, "
            "and inlucdes major, minor, and patch number, "
            "e.g., major.minor.patch."
        ) from None
    # Get installed version
    versions = [
        int(_qd_core.get_version_major()),
        int(_qd_core.get_version_minor()),
        int(_qd_core.get_version_patch()),
    ]
    # Match installed version and required version.
    match = major == versions[0] and (minor < versions[1] or minor == versions[1] and patch <= versions[2])

    if not match:
        raise Exception(
            f"Quadrants version mismatch. Required version >= {major}.{minor}.{patch}, installed version = {_qd_core.get_version_string()}."
        )


def init(
    arch=None,
    default_fp=None,
    default_ip=None,
    _test_mode: bool = False,
    enable_fallback: bool = True,
    require_version: str | None = None,
    print_non_pure: bool = False,
    src_ll_cache: bool = True,
    **kwargs,
):
    """Initializes the Quadrants runtime.

    This should always be the entry point of your Quadrants program. Most
    importantly, it sets the backend used throughout the program.

    Args:
        arch: Backend to use. This is usually :const:`~quadrants.lang.cpu` or :const:`~quadrants.lang.gpu`.
        default_fp (Optional[type]): Default floating-point type.
        default_ip (Optional[type]): Default integral type.
        require_version: A version string.
        print_non_pure: Print the names of kernels, at the time they are executed, which are not annotated with
                        @qd.pure
        src_ll_cache: enable SRC-LL-CACHE, which will accelerate loading from cache, across all architectures,
                      for pure kernels (i.e. kernels declared as @qd.pure)
        **kwargs: Quadrants provides highly customizable compilation through
            ``kwargs``, which allows for fine grained control of Quadrants compiler
            behavior. Below we list some of the most frequently used ones. For a
            complete list, please check out
            https://github.com/taichi-dev/quadrants/blob/master/quadrants/program/compile_config.h.

            * ``cpu_max_num_threads`` (int): Sets the number of threads used by the CPU thread pool.
            * ``debug`` (bool): Enables the debug mode, under which Quadrants does a few more things like boundary checks.
            * ``print_ir`` (bool): Prints the CHI IR of the Quadrants kernels.
            *``offline_cache`` (bool): Enables offline cache of the compiled kernels. Default to True. When this is enabled Quadrants will cache compiled kernel on your local disk to accelerate future calls.
            *``random_seed`` (int): Sets the seed of the random generator. The default is 0.
            *``debug_dump_path`` (str): used as the base path for QD_DUMP_IR and similar
    """
    # FIXME(https://github.com/taichi-dev/quadrants/issues/4811): save the current working directory since it may be
    # changed by the Vulkan backend initialization on OS X.
    current_dir = os.getcwd()

    # Check if installed version meets the requirements.
    if require_version is not None:
        check_require_version(require_version)

    if "default_up" in kwargs:
        raise KeyError("'default_up' is always the unsigned type of 'default_ip'. Please set 'default_ip' instead.")
    # Make a deepcopy in case these args reference to items from qd.cfg, which are
    # actually references. If no copy is made and the args are indeed references,
    # qd.reset() could override the args to their default values.
    default_fp = _deepcopy(default_fp)
    default_ip = _deepcopy(default_ip)
    kwargs = _deepcopy(kwargs)
    reset()

    cfg = impl.default_cfg()
    cfg.offline_cache = True  # Enable offline cache in frontend instead of C++ side

    spec_cfg = _SpecialConfig()
    env_comp = _EnvironmentConfigurator(kwargs, cfg)
    env_spec = _EnvironmentConfigurator(kwargs, spec_cfg)

    # configure default_fp/ip:
    # TODO: move these stuff to _SpecialConfig too:
    env_default_fp = os.environ.get("QD_DEFAULT_FP")
    if env_default_fp:
        if default_fp is not None:
            _qd_core.warn(
                f'Environment variable QD_DEFAULT_FP={env_default_fp} overridden by qd.init argument "default_fp"'
            )
        elif env_default_fp == "32":
            default_fp = f32
        elif env_default_fp == "64":
            default_fp = f64
        elif env_default_fp is not None:
            raise ValueError(f"Invalid QD_DEFAULT_FP={env_default_fp}, should be 32 or 64")

    env_default_ip = os.environ.get("QD_DEFAULT_IP")
    if env_default_ip:
        if default_ip is not None:
            _qd_core.warn(
                f'Environment variable QD_DEFAULT_IP={env_default_ip} overridden by qd.init argument "default_ip"'
            )
        elif env_default_ip == "32":
            default_ip = i32
        elif env_default_ip == "64":
            default_ip = i64
        elif env_default_ip is not None:
            raise ValueError(f"Invalid QD_DEFAULT_IP={env_default_ip}, should be 32 or 64")

    if default_fp is not None:
        impl.get_runtime().set_default_fp(default_fp)
    if default_ip is not None:
        impl.get_runtime().set_default_ip(default_ip)

    # submodule configurations (spec_cfg):
    env_spec.add("log_level", str)
    env_spec.add("gdb_trigger")
    env_spec.add("short_circuit_operators")
    env_spec.add("print_full_traceback")
    env_spec.add("unrolling_limit")

    # compiler configurations (qd.cfg):
    for key in dir(cfg):
        if key in ["arch", "default_fp", "default_ip"]:
            continue
        _cast = type(getattr(cfg, key))
        if _cast is bool:
            _cast = None
        env_comp.add(key, _cast)

    unexpected_keys = kwargs.keys()

    if len(unexpected_keys):
        raise KeyError(f'Unrecognized keyword argument(s) for qd.init: {", ".join(unexpected_keys)}')

    # dispatch configurations that are not in qd.cfg:
    runtime = impl.get_runtime()
    if not _test_mode:
        _qd_core.set_core_trigger_gdb_when_crash(spec_cfg.gdb_trigger)
        runtime.short_circuit_operators = spec_cfg.short_circuit_operators
        runtime.print_full_traceback = spec_cfg.print_full_traceback
        runtime.unrolling_limit = spec_cfg.unrolling_limit
        runtime.src_ll_cache = src_ll_cache
        runtime.print_non_pure = print_non_pure
        _logging.set_logging_level(spec_cfg.log_level.lower())

    # select arch (backend):
    env_arch = os.environ.get("QD_ARCH")
    if env_arch is not None:
        _logging.info(f"Following QD_ARCH setting up for arch={env_arch}")
        arch = _qd_core.arch_from_name(env_arch)
    cfg.arch = adaptive_arch_select(arch, enable_fallback)
    print(f"[Quadrants] Starting on arch={_qd_core.arch_name(cfg.arch)}")

    if cfg.arch == _qd_core.amdgpu and get_os_name() == "win":
        _logging.warn("AMDGPU support on Windows is experimental and may not work as expected.")

    if _test_mode:
        return spec_cfg

    get_default_kernel_profiler().set_kernel_profiler_mode(cfg.kernel_profiler)

    # create a new program:
    impl.get_runtime().create_program()

    _logging.trace("Materializing runtime...")
    impl.get_runtime().prog.materialize_runtime()

    impl._root_fb = _snode.FieldsBuilder()

    if cfg.debug:
        impl.get_runtime()._register_signal_handlers()

    # Recover the current working directory (https://github.com/taichi-dev/quadrants/issues/4811)
    os.chdir(current_dir)
    return None


def no_activate(*args):
    """Deactivates a SNode pointer."""
    compiling_callable = get_runtime().compiling_callable
    assert isinstance(compiling_callable, _qd_core.KernelCxx)
    for v in args:
        compiling_callable.no_activate(v._snode.ptr)


def block_local(*args):
    """Hints Quadrants to cache the fields and to enable the BLS optimization.

    Please visit https://docs.taichi-lang.org/docs/performance
    for how BLS is used.

    Args:
        *args (List[Field]): A list of sparse Quadrants fields.
    """
    if impl.current_cfg().opt_level == 0:
        _logging.warn("""opt_level = 1 is enforced to enable bls analysis.""")
        impl.current_cfg().opt_level = 1
    for a in args:
        for v in a._get_field_members():
            get_runtime().compiling_callable.ast_builder().insert_snode_access_flag(
                _qd_core.SNodeAccessFlag.block_local, v.ptr
            )


def mesh_local(*args):
    """Hints the compiler to cache the mesh attributes
    and to enable the mesh BLS optimization,
    only available for backends supporting `qd.extension.mesh` and to use with mesh-for loop.

    Related to https://github.com/taichi-dev/quadrants/issues/3608

    Args:
        *args (List[Attribute]): A list of mesh attributes or fields accessed as attributes.

    Examples::

        # instantiate model
        mesh_builder = qd.Mesh.tri()
        mesh_builder.verts.place({
            'x' : qd.f32,
            'y' : qd.f32
        })
        model = mesh_builder.build(meta)

        @qd.kernel
        def foo():
            # hint the compiler to cache mesh vertex attribute `x` and `y`.
            qd.mesh_local(model.verts.x, model.verts.y)
            for v0 in model.verts: # mesh-for loop
                for v1 in v0.verts:
                    v0.x += v1.y
    """
    for a in args:
        for v in a._get_field_members():
            get_runtime().compiling_callable.ast_builder().insert_snode_access_flag(
                _qd_core.SNodeAccessFlag.mesh_local, v.ptr
            )


def cache_read_only(*args):
    for a in args:
        for v in a._get_field_members():
            get_runtime().compiling_callable.ast_builder().insert_snode_access_flag(
                _qd_core.SNodeAccessFlag.read_only, v.ptr
            )


def assume_in_range(val, base, low, high):
    """Hints the compiler that a value is between a specified range,
    for the compiler to perform scatchpad optimization, and return the
    value untouched.

    The assumed range is `[base + low, base + high)`.

    Args:

        val (Number): The input value.
        base (Number): The base point for the range interval.
        low (Number): The lower offset relative to `base` (included).
        high (Number): The higher offset relative to `base` (excluded).

    Returns:
        Return the input `value` untouched.

    Example::

        >>> # hint the compiler that x is in range [8, 12).
        >>> x = qd.assume_in_range(x, 10, -2, 2)
        >>> x
        10
    """
    return _qd_core.expr_assume_in_range(
        Expr(val).ptr, Expr(base).ptr, low, high, _qd_core.DebugInfo(impl.get_runtime().get_current_src_info())
    )


def loop_unique(val, covers=None):
    if covers is None:
        covers = []
    if not isinstance(covers, (list, tuple)):
        covers = [covers]
    covers = [x.snode.ptr if isinstance(x, Expr) else x.ptr for x in covers]  # type: ignore
    return _qd_core.expr_loop_unique(
        Expr(val).ptr, covers, _qd_core.DebugInfo(impl.get_runtime().get_current_src_info())
    )


def _parallelize(v):
    """Sets the number of threads to use on CPU."""
    get_runtime().compiling_callable.ast_builder().parallelize(v)
    if v == 1:
        get_runtime().compiling_callable.ast_builder().strictly_serialize()


def _serialize():
    """Sets the number of threads to 1."""
    _parallelize(1)


def _block_dim(dim):
    """Set the number of threads in a block to `dim`."""
    get_runtime().compiling_callable.ast_builder().block_dim(dim)


def _block_dim_adaptive(block_dim_adaptive):
    """Enable/Disable backends set block_dim adaptively."""
    if get_runtime().prog.config().arch != cpu:
        _logging.warn("Adaptive block_dim is supported on CPU backend only")
    else:
        get_runtime().prog.config().cpu_block_dim_adaptive = block_dim_adaptive


def _bit_vectorize():
    """Enable bit vectorization of struct fors on quant_arrays."""
    get_runtime().compiling_callable.ast_builder().bit_vectorize()


def loop_config(
    *,
    block_dim=None,
    serialize=False,
    parallelize=None,
    block_dim_adaptive=True,
    bit_vectorize=False,
):
    """Sets directives for the next loop

    Args:
        block_dim (int): The number of threads in a block on GPU
        serialize (bool): Whether to let the for loop execute serially, `serialize=True` equals to `parallelize=1`
        parallelize (int): The number of threads to use on CPU
        block_dim_adaptive (bool): Whether to allow backends set block_dim adaptively, enabled by default
        bit_vectorize (bool): Whether to enable bit vectorization of struct fors on quant_arrays.

    Examples::

        @qd.kernel
        def break_in_serial_for() -> qd.i32:
            a = 0
            qd.loop_config(serialize=True)
            for i in range(100):  # This loop runs serially
                a += i
                if i == 10:
                    break
            return a

        break_in_serial_for()  # returns 55

        n = 128
        val = qd.field(qd.i32, shape=n)
        @qd.kernel
        def fill():
            qd.loop_config(parallelize=8, block_dim=16)
            # If the kernel is run on the CPU backend, 8 threads will be used to run it
            # If the kernel is run on the CUDA backend, each block will have 16 threads.
            for i in range(n):
                val[i] = i

        u1 = qd.types.quant.int(bits=1, signed=False)
        x = qd.field(dtype=u1)
        y = qd.field(dtype=u1)
        cell = qd.root.dense(qd.ij, (128, 4))
        cell.quant_array(qd.j, 32).place(x)
        cell.quant_array(qd.j, 32).place(y)
        @qd.kernel
        def copy():
            qd.loop_config(bit_vectorize=True)
            # 32 bits, instead of 1 bit, will be copied at a time
            for i, j in x:
                y[i, j] = x[i, j]
    """
    if block_dim is not None:
        _block_dim(block_dim)

    if serialize:
        _parallelize(1)
    elif parallelize is not None:
        _parallelize(parallelize)

    if not block_dim_adaptive:
        _block_dim_adaptive(block_dim_adaptive)

    if bit_vectorize:
        _bit_vectorize()


def global_thread_idx():
    """Returns the global thread id of this running thread,
    only available for cpu and cuda backends.

    For cpu backends this is equal to the cpu thread id,
    For cuda backends this is equal to `block_id * block_dim + thread_id`.

    Example::

        >>> f = qd.field(qd.f32, shape=(16, 16))
        >>> @qd.kernel
        >>> def test():
        >>>     for i in qd.grouped(f):
        >>>         print(qd.global_thread_idx())
        >>>
        test()
    """
    return impl.get_runtime().compiling_callable.ast_builder().insert_thread_idx_expr()


def mesh_patch_idx():
    """Returns the internal mesh patch id of this running thread,
    only available for backends supporting `qd.extension.mesh` and to use within mesh-for loop.

    Related to https://github.com/taichi-dev/quadrants/issues/3608
    """
    return (
        impl.get_runtime()
        .compiling_callable.ast_builder()
        .insert_patch_idx_expr(_qd_core.DebugInfo(impl.get_runtime().get_current_src_info()))
    )


def is_arch_supported(arch):
    """Checks whether an arch is supported on the machine.

    Args:
        arch (quadrants_python.Arch): Specified arch.

    Returns:
        bool: Whether `arch` is supported on the machine.
    """

    arch_table = {
        cuda: _qd_core.with_cuda,
        amdgpu: _qd_core.with_amdgpu,
        metal: _qd_core.with_metal,
        vulkan: _qd_core.with_vulkan,
        cpu: lambda: True,
    }
    with_arch = arch_table.get(arch, lambda: False)
    try:
        return with_arch()
    except Exception as e:
        arch = _qd_core.arch_name(arch)
        _qd_core.warn(
            f"{e.__class__.__name__}: '{e}' occurred when detecting "
            f"{arch}, consider adding `QD_ENABLE_{arch.upper()}=0` "
            f" to environment variables to suppress this warning message."
        )
        return False


def adaptive_arch_select(arch, enable_fallback):
    if arch is None:
        return cpu
    if not isinstance(arch, (list, tuple)):
        arch = [arch]
    for a in arch:
        if is_arch_supported(a):
            return a
    if not enable_fallback:
        raise RuntimeError(f"Arch={arch} is not supported")
    _logging.warn(f"Arch={arch} is not supported, falling back to CPU")
    return cpu


def get_host_arch_list():
    return [_qd_core.host_arch()]


def is_extension_enabled(ext: Extension) -> bool:
    """
    Directly returns whether extension is enabled, without needing to
    pass in current architecture. Also takes into account config, in the case
    of adstack.
    """
    arch = impl.current_cfg().arch
    if ext == extension.adstack:
        return is_extension_supported(arch, ext) and impl.current_cfg().ad_stack_experimental_enabled
    return is_extension_supported(arch, ext)


def dump_compile_config() -> None:
    """
    Dumps the compile config, which can be usful for example in diagnosing fastcache issues, since
    the fastcache cache keys depend on the compile config.
    """
    config = impl.current_cfg()
    config_l = []
    for _k in sorted(dir(config)):
        if _k.startswith("_"):
            continue
        v = getattr(config, _k)
        config_l.append(f"{_k}={v}")
    print("\n".join(config_l))


__all__ = [
    "i",
    "ij",
    "ijk",
    "ijkl",
    "ijl",
    "ik",
    "ikl",
    "il",
    "j",
    "jk",
    "jkl",
    "jl",
    "k",
    "kl",
    "l",
    "x86_64",
    "x64",
    "arm64",
    "cpu",
    "cuda",
    "amdgpu",
    "gpu",
    "metal",
    "vulkan",
    "extension",
    "loop_config",
    "global_thread_idx",
    "assume_in_range",
    "block_local",
    "cache_read_only",
    "dump_compile_config",
    "init",
    "mesh_local",
    "no_activate",
    "reset",
    "mesh_patch_idx",
    "is_extension_enabled",
]
