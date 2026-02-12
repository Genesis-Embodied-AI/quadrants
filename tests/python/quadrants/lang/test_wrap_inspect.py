import importlib
import sys
import textwrap

from quadrants._test_tools import textwrap2
from quadrants.lang import _wrap_inspect

from tests import test_utils


@test_utils.test()
def test_wrap_inspect() -> None:
    test_files_path = "tests/python/quadrants/lang/fast_caching/test_files"
    if test_files_path not in sys.path:
        sys.path.append(test_files_path)

    wrap_inspect = importlib.import_module("wrap_inspect")
    source_info, src = _wrap_inspect.get_source_info_and_src(wrap_inspect.my_func)
    assert source_info.function_name == "my_func"
    assert source_info.start_lineno == 2
    assert source_info.end_lineno == 5
    assert len(src) == 4
    assert all([line != "" and len(line) > 2 for line in src])
    actual_src = "".join(src)
    expected = textwrap.dedent(
        """\
    def my_func(a: int, b: int) -> int:
        a += 5
        b += a
        return a + b
    """
    )
    assert actual_src == expected

    source_info, src = _wrap_inspect.get_source_info_and_src(wrap_inspect.Foo.my_func2)
    assert source_info.function_name == "my_func2"
    assert source_info.start_lineno == 10
    assert source_info.end_lineno == 13
    assert len(src) == 4
    assert all([line != "" and len(line) > 2 for line in src])
    actual_src = "".join(src)
    expected = textwrap2.dedent(
        4,
        """\
        def my_func2(self, a: int, b: int) -> int:
            a += 5
            b += a
            return a + b
    """,
    )
    assert actual_src == expected

    sys.path.remove(test_files_path)
