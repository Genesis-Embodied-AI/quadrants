import pytest

import quadrants as qd

from tests import test_utils

# TODO: validation layer support on macos vulkan backend is not working.
vk_on_mac = (qd.vulkan, "Darwin")

# TODO: capfd doesn't function well on CUDA backend on Windows
cuda_on_windows = (qd.cuda, "Windows")


def filter_lines(target: str, match: str) -> str:
    """
    Returns target string, with
    - only lines included that contains `match` string
        - this is so we can filter out various other stdout messages
    - anything before `match` string is removed
        - this is so we can handle Vulkan print messages, which are often prefixed with something like
          `vkSubmitQueue():  `
    """
    lines = []
    for line in target.split("\n"):
        if match in line:
            _, splitter, post = line.partition(match)
            lines.append(f"{splitter}{post}")
    return "\n".join(lines)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_docs_scalar_self_documenting_exp(capfd):
    a = qd.field(qd.f32, 4)

    @qd.kernel
    def func():
        a[0] = 1.0

        # with self-documenting expressions
        print(f"TEST_PRINT: {a[0] = :.1f}")

    func()
    qd.sync()

    out, err = capfd.readouterr()
    out = filter_lines(out, "TEST_PRINT:")
    expected_out = """TEST_PRINT: a[0] = 1.0"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_docs_matrix_self_documenting_exp(capfd):
    @qd.kernel
    def func():
        m = qd.Matrix([[2e1, 3e2, 4e3], [5e4, 6e5, 7e6]], qd.f32)

        # with self-documenting expressions
        print(f"TEST_PRINT: {m = :g}")

    func()
    qd.sync()

    out, err = capfd.readouterr()
    out = filter_lines(out, "TEST_PRINT:")
    expected_out = """TEST_PRINT: m = [[20, 300, 4000], [50000, 600000, 7e+06]]"""
    assert out == expected_out and err == ""


# Not really testable..
# Just making sure it does not crash
# Metal doesn't support print() or 64-bit data
@pytest.mark.parametrize("dt", qd.types.primitive_types.all_types)
@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_print(dt):
    @qd.kernel
    def func():
        print(qd.cast(123.4, dt))

    func()
    # Discussion: https://github.com/taichi-dev/quadrants/issues/1063#issuecomment-636421904
    # Synchronize to prevent cross-test failure of print:
    qd.sync()


# TODO: As described by @k-ye above, what we want to ensure
#       is that, the content shows on console is *correct*.
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_multi_print():
    @qd.kernel
    def func(x: qd.i32, y: qd.f32):
        print(x, 1234.5, y)

    func(666, 233.3)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_string():
    @qd.kernel
    def func(x: qd.i32, y: qd.f32):
        # make sure `%` doesn't break vprintf:
        print("hello, world! %s %d %f", 233, y)
        print("cool", x, "well", y)

    func(666, 233.3)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_matrix():
    x = qd.Matrix.field(2, 3, dtype=qd.f32, shape=())
    y = qd.Vector.field(3, dtype=qd.f32, shape=3)

    @qd.kernel
    def func(k: qd.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print("hello", x[None], "world!")
        print(y[2] * k, x[None] / k, y[2])

    func(233.3)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_matrix_string_format():
    x = qd.Matrix.field(2, 3, dtype=qd.f32, shape=())
    y = qd.Vector.field(3, dtype=qd.f32, shape=3)

    @qd.kernel
    def func(k: qd.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print("hello {} world!".format(x[None]))
        print("{} {} {}".format(y[2] * k, x[None] / k, y[2]))

    func(233.3)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_matrix_string_format_with_spec(capfd):
    x = qd.Matrix.field(2, 3, dtype=qd.f32, shape=())
    y = qd.Vector.field(3, dtype=qd.f32, shape=3)
    z = qd.Matrix.field(2, 3, dtype=qd.i32, shape=())

    @qd.kernel
    def func(k: qd.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print("TEST_PRINT: hello {:.2f} world!".format(x[None]))
        print("TEST_PRINT: {:.3f} {:e} {:.2}".format(y[2] * k, x[None] / k, y[2]))
        print("TEST_PRINT: hello {:.10d} world!".format(z[None]))

    func(233.3)
    qd.sync()

    out, err = capfd.readouterr()
    out = filter_lines(out, "TEST_PRINT: ")
    expected_out = """TEST_PRINT: hello [[-1.00, 0.00, 0.00], [0.00, 0.00, 0.00]] world!
TEST_PRINT: [233.300, 233.300, 233.300] [[-4.286326e-03, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]] [1.00, 1.00, 1.00]
TEST_PRINT: hello [[0000000000, 0000000000, 0000000000], [0000000000, 0000000000, 0000000000]] world!"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_matrix_string_format_with_spec_mismatch():
    x = qd.Matrix.field(2, 3, dtype=qd.f32, shape=())
    y = qd.Vector.field(3, dtype=qd.f32, shape=3)
    z = qd.Matrix.field(2, 3, dtype=qd.i32, shape=())

    @qd.kernel
    def test_x():
        print("hello {:.2d} world!".format(x[None]))

    @qd.kernel
    def test_y(k: qd.f32):
        print("{:- #0.233lli} {:e} {:.2}".format(y[2] * k, x[None] / k, y[2]))

    @qd.kernel
    def test_z():
        print("hello {:.2e} world!".format(z[None]))

    x[None][0, 0] = -1.0
    y[2] += 1.0
    with pytest.raises(qd.QuadrantsTypeError, match=r"'.2d' doesn't match 'f32'."):
        test_x()
    with pytest.raises(qd.QuadrantsTypeError, match=r"'- #0.233lli' doesn't match 'f32'."):
        test_y(233.3)
    with pytest.raises(qd.QuadrantsTypeError, match=r"'.2e' doesn't match 'i32'."):
        test_z()
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_matrix_fstring():
    x = qd.Matrix.field(2, 3, dtype=qd.f32, shape=())
    y = qd.Vector.field(3, dtype=qd.f32, shape=3)

    @qd.kernel
    def func(k: qd.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print(f"hello {x[None]} world!")
        print(f"{y[2] * k} {x[None] / k} {y[2]}")

    func(233.3)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_matrix_fstring_with_spec(capfd):
    x = qd.Matrix.field(2, 3, dtype=qd.f32, shape=())
    y = qd.Vector.field(3, dtype=qd.f32, shape=3)
    z = qd.Matrix.field(2, 3, dtype=qd.i32, shape=())

    @qd.kernel
    def func(k: qd.f32):
        x[None][0, 0] = -1.0
        y[2] += 1.0
        print(f"TEST_PRINT: hello {x[None]:.2f} world!")
        print(f"TEST_PRINT: {(y[2] * k):.3f} {(x[None] / k):e} {y[2]:.2}")
        print(f"TEST_PRINT: hello {z[None]:.2d} world!")

    func(233.3)
    qd.sync()

    out, err = capfd.readouterr()
    out = filter_lines(out, "TEST_PRINT: ")
    expected_out = """TEST_PRINT: hello [[-1.00, 0.00, 0.00], [0.00, 0.00, 0.00]] world!
TEST_PRINT: [233.300, 233.300, 233.300] [[-4.286326e-03, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]] [1.00, 1.00, 1.00]
TEST_PRINT: hello [[00, 00, 00], [00, 00, 00]] world!"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_matrix_fstring_with_spec_mismatch():
    x = qd.Matrix.field(2, 3, dtype=qd.f32, shape=())
    y = qd.Vector.field(3, dtype=qd.f32, shape=3)
    z = qd.Matrix.field(2, 3, dtype=qd.i32, shape=())

    @qd.kernel
    def test_x():
        print(f"hello {x[None]:.2d} world!")

    @qd.kernel
    def test_y(k: qd.f32):
        print(f"{(y[2] * k):- #0.233lli} {(x[None] / k):e} {y[2]:.2}")

    @qd.kernel
    def test_z():
        print(f"hello {z[None]:.2e} world!")

    x[None][0, 0] = -1.0
    y[2] += 1.0
    with pytest.raises(qd.QuadrantsTypeError, match=r"'.2d' doesn't match 'f32'."):
        test_x()
    with pytest.raises(qd.QuadrantsTypeError, match=r"'- #0.233lli' doesn't match 'f32'."):
        test_y(233.3)
    with pytest.raises(qd.QuadrantsTypeError, match=r"'.2e' doesn't match 'i32'."):
        test_z()
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_docs_scalar(capfd):
    a = qd.field(qd.f32, 4)

    @qd.kernel
    def func():
        a[0] = 1.0

        # comma-separated string
        print("TEST_PRINT: a[0] =", a[0])

        # f-string
        print(f"TEST_PRINT: a[0] = {a[0]}")
        # with format specifier
        print(f"TEST_PRINT: a[0] = {a[0]:.1f}")
        # without conversion
        print(f"TEST_PRINT: a[0] = {a[0]:.1}")

        # formatted string via `str.format()` method
        print("TEST_PRINT: a[0] = {}".format(a[0]))
        # with format specifier
        print("TEST_PRINT: a[0] = {:.1f}".format(a[0]))
        # without conversion
        print("TEST_PRINT: a[0] = {:.1}".format(a[0]))
        # with positional arguments
        print(
            "TEST_PRINT: a[3] = {3:.3f}, a[2] = {2:.2f}, a[1] = {1:.1f}, a[0] = {0:.0f}".format(a[0], a[1], a[2], a[3])
        )

    func()
    qd.sync()

    out, err = capfd.readouterr()
    out = filter_lines(out, "TEST_PRINT: ")
    expected_out = """TEST_PRINT: a[0] = 1.000000
TEST_PRINT: a[0] = 1.000000
TEST_PRINT: a[0] = 1.0
TEST_PRINT: a[0] = 1.0
TEST_PRINT: a[0] = 1.000000
TEST_PRINT: a[0] = 1.0
TEST_PRINT: a[0] = 1.0
TEST_PRINT: a[3] = 0.000, a[2] = 0.00, a[1] = 0.0, a[0] = 1"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_docs_matrix(capfd):
    a = qd.field(qd.f32, 4)

    @qd.kernel
    def func():
        m = qd.Matrix([[2e1, 3e2, 4e3], [5e4, 6e5, 7e6]], qd.f32)

        # comma-seperated string is supported
        print("TEST_PRINT: m =", m)

        # f-string is supported
        print(f"TEST_PRINT: m = {m}")
        # can with format specifier
        print(f"TEST_PRINT: m = {m:.1f}")
        # can omitting conversion
        print(f"TEST_PRINT: m = {m:.1}")

        # formatted string via `str.format()` method is supported
        print("TEST_PRINT: m = {}".format(m))
        # can with format specifier
        print("TEST_PRINT: m = {:e}".format(m))
        # and can omitting conversion
        print("TEST_PRINT: m = {:.1}".format(m))

    func()
    qd.sync()

    out, err = capfd.readouterr()
    out = filter_lines(out, "TEST_PRINT: ")
    expected_out = """TEST_PRINT: m = [[20.000000, 300.000000, 4000.000000], [50000.000000, 600000.000000, 7000000.000000]]
TEST_PRINT: m = [[20.000000, 300.000000, 4000.000000], [50000.000000, 600000.000000, 7000000.000000]]
TEST_PRINT: m = [[20.0, 300.0, 4000.0], [50000.0, 600000.0, 7000000.0]]
TEST_PRINT: m = [[20.0, 300.0, 4000.0], [50000.0, 600000.0, 7000000.0]]
TEST_PRINT: m = [[20.000000, 300.000000, 4000.000000], [50000.000000, 600000.000000, 7000000.000000]]
TEST_PRINT: m = [[2.000000e+01, 3.000000e+02, 4.000000e+03], [5.000000e+04, 6.000000e+05, 7.000000e+06]]
TEST_PRINT: m = [[20.0, 300.0, 4000.0], [50000.0, 600000.0, 7000000.0]]"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_sep_end():
    @qd.kernel
    def func():
        # hello 42 world!
        print("hello", 42, "world!")
        # hello 42 Quadrants 233 world!
        print("hello", 42, "Tai", end="")
        print("chi", 233, "world!")
        # hello42world!
        print("hello", 42, "world!", sep="")
        # '  ' (with no newline)
        print("  ", end="")
        # 'helloaswd42qwer'
        print("  ", 42, sep="aswd", end="qwer")

    func()
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_multiple_threads():
    x = qd.field(dtype=qd.f32, shape=(128,))

    @qd.kernel
    def func(k: qd.f32):
        for i in x:
            x[i] = i * k
            print("x[", i, "]=", x[i])

    func(0.1)
    qd.sync()
    func(10.0)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_list():
    x = qd.Matrix.field(2, 3, dtype=qd.f32, shape=(2, 3))
    y = qd.Vector.field(3, dtype=qd.f32, shape=())

    @qd.kernel
    def func(k: qd.f32):
        w = [k, x.shape]
        print(w + [y.n])  # [233.3, [2, 3], 3]
        print(x.shape)  # [2, 3]
        print(y.shape)  # []
        z = (1,)
        print([1, k**2, k + 1])  # [1, 233.3, 234.3]
        print(z)  # [1]
        print([y[None], z])  # [[0, 0, 0], [1]]
        print([])  # []

    func(233.3)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_python_scope_print_field():
    x = qd.Matrix.field(2, 3, dtype=qd.f32, shape=())
    y = qd.Vector.field(3, dtype=qd.f32, shape=3)
    z = qd.field(dtype=qd.f32, shape=3)

    print(x)
    print(y)
    print(z)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_string_format():
    @qd.kernel
    def func(k: qd.f32):
        print(123)
        print("{} abc".format(123))
        print("{} {} {}".format(1, 2, 3))
        print("{} {name} {value}".format(k, name=999, value=123))
        name = 123.4
        value = 456.7
        print("{} {name} {value}".format(k, name=name, value=value))

    func(233.3)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_string_format_with_spec(capfd):
    @qd.kernel
    def func(k: qd.f32):
        print("TEST_PRINT: ", 123)
        print("TEST_PRINT: {:d} abc".format(123))
        print("TEST_PRINT: {:i} {:.1} {:.10d}".format(1, 2, 3))
        print("TEST_PRINT: {:.2} {name:i} {value:d}".format(k, name=999, value=123))
        name = 123.4
        value = 456.7
        print("TEST_PRINT: {:.2e} {name:.3G} {value:.4f}".format(k, name=name, value=value))

    func(233.3)
    qd.sync()
    out, err = capfd.readouterr()
    out = filter_lines(out, "TEST_PRINT: ")
    expected_out = """TEST_PRINT:  123
TEST_PRINT: 123 abc
TEST_PRINT: 1 2 0000000003
TEST_PRINT: 233.30 999 123
TEST_PRINT: 2.33e+02 123 456.7000"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_string_format_with_spec_mismatch():
    @qd.func
    def foo1(x):
        return x + 1

    @qd.kernel
    def test_i(i: qd.i32):
        print("{:u}".format(foo1(i)))

    @qd.kernel
    def test_u(u: qd.u32):
        print("{:d}".format(foo1(u)))

    @qd.kernel
    def test_f(u: qd.f32):
        print("{:i}".format(foo1(u)))

    with pytest.raises(qd.QuadrantsTypeError, match=r"'u' doesn't match 'i32'."):
        test_i(123)
    with pytest.raises(qd.QuadrantsTypeError, match=r"'d' doesn't match 'u32'."):
        test_u(123)
    with pytest.raises(qd.QuadrantsTypeError, match=r"'i' doesn't match 'f32'."):
        test_f(123)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_string_format_with_positional_arg(capfd):
    @qd.kernel
    def func(k: qd.f32):
        print("TEST_PRINT: {0} {1} {2}".format(1, 2, 3))
        print("TEST_PRINT: {2} {1} {}".format(3, 2, 1))
        print("TEST_PRINT: {2} {} {1} {k} {0} {k} {0} {k}".format(3, 2, 1, k=k))

    func(233.3)
    qd.sync()
    out, err = capfd.readouterr()
    out = filter_lines(out, "TEST_PRINT: ")
    expected_out = """TEST_PRINT: 1 2 3
TEST_PRINT: 1 2 3
TEST_PRINT: 1 3 2 233.300003 3 233.300003 3 233.300003"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_string_format_with_positional_arg_with_spec(capfd):
    @qd.kernel
    def func(k: qd.f32):
        print("TEST_PRINT: {0:d} {1:} {2:i}".format(1, 2, 3))
        print("TEST_PRINT: {2:d} {1:.2} {:.10}".format(3, 2, 1))
        print("TEST_PRINT: {2:.1} {:.2} {1:.3} {k:.4e} {0:.5} {k:.5f} {0:.5} {k:.4g}".format(3.0, 2.0, 1.0, k=k))

    func(233.3)
    qd.sync()
    out, err = capfd.readouterr()
    out = filter_lines(out, "TEST_PRINT: ")
    expected_out = """TEST_PRINT: 1 2 3
TEST_PRINT: 1 02 0000000003
TEST_PRINT: 1.0 3.00 2.000 2.3330e+02 3.00000 233.30000 3.00000 233.3"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_string_format_with_positional_arg_mismatch():
    @qd.kernel
    def func(k: qd.f32):
        print("{0} {1} {2}".format(1, 2))
        print("{2} {1} {}".format(3, 2, 1))
        print("{0} {} {0} {k} {0} {k}".format(1, k=k))

    @qd.kernel
    def func_k_not_used(k: qd.f32):
        print("".format(k=k))

    @qd.kernel
    def func_k_not_defined():
        print("{k}".format())

    @qd.kernel
    def func_more_args():
        print("{0} {1} {2}".format(1, 2, 3, 4))

    @qd.kernel
    def func_less_args():
        print("{0} {1} {2}".format(1, 2))

    with pytest.raises(
        qd.QuadrantsSyntaxError,
        match=r"Expected 3 positional argument\(s\), but received 4 instead.",
    ):
        func_more_args()
    with pytest.raises(
        qd.QuadrantsSyntaxError,
        match=r"Expected 3 positional argument\(s\), but received 2 instead.",
    ):
        func_less_args()
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"Keyword 'k' not used."):
        func_k_not_used(233.3)
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"Keyword 'k' not found."):
        func_k_not_defined()
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_fstring():
    @qd.func
    def foo1(x):
        return x + 1

    @qd.kernel
    def func(i: qd.i32, f: qd.f32):
        print(f"qwe {foo1(1)} {foo1(2) * 2 - 1} {i} {f} {4} {True} {1.23}")

    func(123, 4.56)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_fstring_with_spec(capfd):
    @qd.func
    def foo1(x):
        return x + 1

    @qd.kernel
    def func(i: qd.i32, f: qd.f32):
        print(f"TEST_PRINT: qwe {foo1(1):d} {(foo1(2) * 2 - 1):.10d} {i} {f:.1f} {4} {True} {1.23}")

    func(123, 4.56)
    qd.sync()
    out, err = capfd.readouterr()
    out = filter_lines(out, "TEST_PRINT: ")
    expected_out = """TEST_PRINT: qwe 2 0000000005 123 4.6 4 True 1.23"""
    assert out == expected_out and err == ""


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_fstring_with_spec_mismatch():
    @qd.func
    def foo1(x):
        return x + 1

    @qd.kernel
    def test_i(i: qd.i32):
        print(f"{foo1(i):u}")

    @qd.kernel
    def test_u(u: qd.u32):
        print(f"{foo1(u):d}")

    @qd.kernel
    def test_f(u: qd.f32):
        print(f"{foo1(u):i}")

    with pytest.raises(qd.QuadrantsTypeError, match=r"'u' doesn't match 'i32'."):
        test_i(123)
    with pytest.raises(qd.QuadrantsTypeError, match=r"'d' doesn't match 'u32'."):
        test_u(123)
    with pytest.raises(qd.QuadrantsTypeError, match=r"'i' doesn't match 'f32'."):
        test_f(123)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_u64():
    @qd.kernel
    def func(i: qd.u64):
        print("i =", i)

    func(2**64 - 1)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac], debug=True)
def test_print_i64():
    @qd.kernel
    def func(i: qd.i64):
        print("i =", i)

    func(-(2**63) + 2**31)
    qd.sync()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac, cuda_on_windows], debug=True)
def test_print_seq(capfd):
    @qd.kernel
    def foo():
        print("inside kernel")

    foo()
    print("outside kernel")
    out = capfd.readouterr().out
    assert "inside kernel\noutside kernel" in out


@test_utils.test(arch=[qd.cpu, qd.cuda], print_ir=True, debug=True)
def test_fp16_print_ir():
    half2 = qd.types.vector(n=2, dtype=qd.f16)

    @qd.kernel
    def test():
        x = half2(1.0)
        y = half2(2.0)

        for i in range(2):
            x[i] = y[i]
            print(x[i])

    test()
