from inspect import currentframe, getframeinfo

import pytest

import quadrants as ti

from tests import test_utils


@test_utils.test(print_full_traceback=False)
def test_exception_multiline():
    frameinfo = getframeinfo(currentframe())
    with pytest.raises(ti.QuadrantsNameError) as e:
        # yapf: disable
        @ti.kernel
        def foo():
            aaaa(111,
                 1211222,

                 23)
        foo()
        # yapf: enable

    msg = f"""
File "{frameinfo.filename}", line {frameinfo.lineno + 5}, in foo:
            aaaa(111,
            ^^^^"""
    assert msg in e.value.args[0]


@test_utils.test(print_full_traceback=False)
def test_exception_from_func():
    frameinfo = getframeinfo(currentframe())
    with pytest.raises(ti.QuadrantsNameError) as e:

        @ti.func
        def baz():
            t()

        @ti.func
        def bar():
            baz()

        @ti.kernel
        def foo():
            bar()

        foo()
    lineno = frameinfo.lineno
    file = frameinfo.filename
    msg_l = [
        f"""File "{file}", line {lineno + 13}, in foo:
            bar()
            ^^^^^""",
        f"""File "{file}", line {lineno + 9}, in bar:
            baz()
            ^^^^^""",
        f"""File "{file}", line {lineno + 5}, in baz:
            t()
            ^""",
    ]
    for msg in msg_l:
        assert msg in e.value.args[0]


@test_utils.test(print_full_traceback=False)
def test_tab():
    frameinfo = getframeinfo(currentframe())
    with pytest.raises(ti.QuadrantsNameError) as e:
        # yapf: disable
        @ti.kernel
        def foo():
            a(11,	22,	3)
        foo()
        # yapf: enable
    lineno = frameinfo.lineno
    file = frameinfo.filename
    msg = f"""
File "{file}", line {lineno + 5}, in foo:
            a(11,   22, 3)
            ^"""
    assert msg in e.value.args[0]


@test_utils.test(print_full_traceback=False)
def test_super_long_line():
    frameinfo = getframeinfo(currentframe())
    with pytest.raises(ti.QuadrantsNameError) as e:
        # yapf: disable
        @ti.kernel
        def foo():
            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(111)
        foo()
        # yapf: enable
    lineno = frameinfo.lineno
    file = frameinfo.filename
    msg = f"""
File "{file}", line {lineno + 5}, in foo:
            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbaaaaaa
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bbbbbbbbbbbbbbbbbbbbbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(111)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"""
    assert msg in e.value.args[0]


@test_utils.test(print_full_traceback=False)
def test_exception_in_node_with_body():
    frameinfo = getframeinfo(currentframe())
    @ti.kernel
    def foo():
        for i in range(1, 2, 3):
            a = 1
            b = 1
            c = 1
            d = 1

    with pytest.raises(ti.QuadrantsCompilationError) as e:
        foo()
    lineno = frameinfo.lineno
    file = frameinfo.filename
    msg = f"""
File "{file}", line {lineno + 3}, in foo:
        for i in range(1, 2, 3):
        ^^^^^^^^^^^^^^^^^^^^^^^^
Range should have 1 or 2 arguments, found 3"""
    assert msg in e.value.args[0]

