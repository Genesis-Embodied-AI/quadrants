import importlib.util
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


def import_kernel_from_file(kernel_filepath: Path, kernel_name: str):
    spec = importlib.util.spec_from_file_location(kernel_name, kernel_filepath)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[kernel_name] = module
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return getattr(module, kernel_name)


@contextmanager
def load_kernel_from_string(kernel_str: str, kernel_name: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / f"{kernel_name}.py"
        with open(filepath, "w") as f:
            f.write(kernel_str)
        try:
            kernel = import_kernel_from_file(kernel_filepath=filepath, kernel_name=kernel_name)
            yield kernel
        finally:
            if kernel_name in sys.modules:
                del sys.modules[kernel_name]
