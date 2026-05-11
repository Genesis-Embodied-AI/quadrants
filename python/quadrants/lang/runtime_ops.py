# type: ignore

from quadrants.lang import impl


def sync():
    """Synchronizes the default stream.

    Blocks the calling thread until all work on the default GPU stream has completed.  Kernels launched on explicit
    streams created via :func:`quadrants.create_stream` are **not** waited on — call ``stream.synchronize()`` for those.
    """
    impl.get_runtime().sync()


__all__ = ["sync"]
