import quadrants as ti


@ti.kernel
def lcg_ti(B: int, lcg_its: int, a: ti.types.NDArray[ti.i32, 1]) -> None:
    """
    Linear congruential generator https://en.wikipedia.org/wiki/Linear_congruential_generator
    """
    for i in range(B):
        x = a[i]
        for j in range(lcg_its):
            x = (1664525 * x + 1013904223) % 2147483647
        a[i] = x


def main() -> None:
    ti.init(arch=ti.cpu)

    B = 10
    lcg_its = 10

    a = ti.ndarray(ti.int32, (B,))

    lcg_ti(B, lcg_its, a)
    print(f"LCG for B={B}, lcg_its={lcg_its}: ", a.to_numpy())  # pylint: disable=no-member


main()
