import time

import quadrants as qd


@qd.kernel
def lcg_ti(B: int, lcg_its: int, a: qd.types.NDArray[qd.i32, 1]) -> None:
    for i in range(B):
        x = a[i]
        for j in range(lcg_its):
            x = (1664525 * x + 1013904223) % 2147483647
        a[i] = x


def main() -> None:
    qd.init(arch=qd.gpu)

    B = 16000
    a = qd.ndarray(qd.int32, (B,))

    qd.sync()
    start = time.time()
    lcg_ti(B, 1000, a)
    qd.sync()
    end = time.time()
    print("elapsed", end - start)

    # [Quadrants] version 1.8.0, llvm 15.0.7, commit 5afed1c9, osx, python 3.10.16
    # [Quadrants] Starting on arch=metal
    # elapsed 0.04660296440124512
    # (on mac air m4)


main()
