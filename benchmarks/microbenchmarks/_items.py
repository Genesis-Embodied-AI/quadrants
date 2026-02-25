import quadrants as qd
from microbenchmarks._utils import size2tag


class BenchmarkItem:
    name = "item"

    def __init__(self):
        self._items = {}  # {'tag': impl, ...}

    def get(self):
        return self._items

    def get_tags(self):
        return list(self._items.keys())

    def impl(self, tag: str):
        return self._items[tag]

    def remove(self, tags: list):
        for tag in tags:
            self._items.pop(tag)

    def update(self, adict: dict):
        self._items.update(adict)


class DataType(BenchmarkItem):
    name = "dtype"
    integer_list = ["i32", "i64"]

    def __init__(self):
        self._items = {
            str(qd.i32): qd.i32,
            str(qd.i64): qd.i64,
            str(qd.f32): qd.f32,
            str(qd.f64): qd.f64,
        }

    def remove_integer(self):
        self.remove(self.integer_list)

    @staticmethod
    def is_integer(dtype: str):
        integer_list = ["i32", "u32", "i64", "u64"]
        return True if dtype in integer_list else False


class DataSize(BenchmarkItem):
    name = "dsize"

    def __init__(self):
        self._items = {}
        for i in range(2, 10, 2):  # [16KB,256KB,4MB,64MB]
            size_bytes = (4**i) * 1024  # kibibytes(KiB) = 1024
            self._items[size2tag(size_bytes)] = size_bytes


class Container(BenchmarkItem):
    name = "container"

    def __init__(self):
        self._items = {"field": qd.field, "ndarray": qd.ndarray}


class MathOps(BenchmarkItem):
    name = "math_op"

    # reference: https://docs.taichi-lang.org/docs/operator
    def __init__(self):
        self._items = {
            # Trigonometric
            "sin": qd.sin,
            "cos": qd.cos,
            "tan": qd.tan,
            "asin": qd.asin,
            "acos": qd.acos,
            "tanh": qd.tanh,
            # Other arithmetic
            "sqrt": qd.sqrt,
            "rsqrt": qd.rsqrt,  # A fast version for `1 / qd.sqrt(x)`.
            "exp": qd.exp,
            "log": qd.log,
            "round": qd.round,
            "floor": qd.floor,
            "ceil": qd.ceil,
            "abs": qd.abs,
        }


class AtomicOps(BenchmarkItem):
    name = "atomic_op"

    def __init__(self):
        self._items = {
            "atomic_add": qd.atomic_add,
            "atomic_sub": qd.atomic_sub,
            "atomic_and": qd.atomic_and,
            "atomic_or": qd.atomic_or,
            "atomic_xor": qd.atomic_xor,
            "atomic_max": qd.atomic_max,
            "atomic_min": qd.atomic_min,
        }

    @staticmethod
    def is_logical_op(op: str):
        logical_op_list = ["atomic_and", "atomic_or", "atomic_xor"]
        return True if op in logical_op_list else False

    @staticmethod
    def is_supported_type(op: str, dtype: str):
        if AtomicOps.is_logical_op(op) and not DataType.is_integer(dtype):
            return False
        else:
            return True
