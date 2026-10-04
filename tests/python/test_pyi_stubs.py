import subprocess
import sys

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_pyi_stubs(tmpdir):
    test_code = """
import quadrants._lib.core.quadrants_python
reveal_type(quadrants._lib.core.quadrants_python)
"""
    test_file = tmpdir / "tmp_mypy_test.py"
    test_file.write(test_code)

    res = subprocess.check_output([sys.executable, "-m", "pyright", test_file]).decode("utf-8")
    assert "unknown" not in res.lower()


@test_utils.test(arch=qd.cuda)
def test_cuda_event_query_pyi(tmpdir):
    test_code = """
from quadrants._lib.core.quadrants_python import Program

def query_event(prog: Program, event_handle: int) -> int:
    return prog._cuda_event_query(event_handle)
"""
    test_file = tmpdir / "cuda_event_query.py"
    test_file.write(test_code)

    subprocess.check_call([sys.executable, "-m", "pyright", "--pythonpath", sys.executable, str(test_file)])
