# from 2 to 5, 4 lines
def my_func(a: int, b: int) -> int:
    a += 5
    b += a
    return a + b


class Foo:
    # from 10 to 13, 4 lines
    def my_func2(self, a: int, b: int) -> int:
        a += 5
        b += a
        return a + b
