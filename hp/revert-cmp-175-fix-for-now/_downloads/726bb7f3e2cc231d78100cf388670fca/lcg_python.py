import time

import numpy as np
import numpy.typing as npt


def lcg_np(B: int, lcg_its: int, a: npt.NDArray) -> None:
    for i in range(B):
        x = a[i]
        for j in range(lcg_its):
            x = (1664525 * x + 1013904223) % 2147483647
        a[i] = x


def main() -> None:
    B = 16000
    a = np.ndarray((B,), np.int32)

    start = time.time()
    lcg_np(B, 1000, a)
    end = time.time()
    print("elapsed", end - start)
    # elapsed 5.552601099014282 on macbook air m4


main()
