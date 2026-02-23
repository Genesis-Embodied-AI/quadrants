import quadrants as qd


@qd.kernel
def lcg_ti(B: int, lcg_its: int, a: qd.types.NDArray[qd.i32, 1]) -> None:
    """
    Linear congruential generator https://en.wikipedia.org/wiki/Linear_congruential_generator
    """
    for i in range(B):
        x = a[i]
        for j in range(lcg_its):
            x = (1664525 * x + 1013904223) % 2147483647
        a[i] = x


def main() -> None:
    qd.init(arch=qd.cpu)

    B = 10
    lcg_its = 10

    a = qd.ndarray(qd.int32, (B,))

    lcg_ti(B, lcg_its, a)
    print(f"LCG for B={B}, lcg_its={lcg_its}: ", a.to_numpy())  # pylint: disable=no-member


main()
