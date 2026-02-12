# type: ignore

from quadrants.lang import impl


def sync():
    """Blocks the calling thread until all the previously
    launched Quadrants kernels have completed.
    """
    impl.get_runtime().sync()


__all__ = ["sync"]
