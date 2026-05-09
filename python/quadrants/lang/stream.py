import weakref
from contextlib import contextmanager

from quadrants.lang import impl


def _get_prog_weakref():
    return weakref.ref(impl.get_runtime().prog)


class Stream:
    """Wraps a backend-specific GPU stream for concurrent kernel execution.

    On backends without native streams (e.g. CPU), this is a no-op object. Call destroy() explicitly or use as
    a context manager to ensure cleanup.
    """

    def __init__(self, handle: int, prog_ref: weakref.ref | None = None):
        self._handle = handle
        self._prog_ref = prog_ref

    @property
    def handle(self) -> int:
        return self._handle

    def _prog(self):
        """Resolve the owning Program, or None if the owner was collected."""
        if self._prog_ref is not None:
            return self._prog_ref()
        return impl.get_runtime().prog

    def synchronize(self):
        """Block until all operations on this stream complete."""
        prog = self._prog()
        if prog is None:
            raise RuntimeError("Stream's owning Program has been destroyed (e.g. after qd.reset())")
        prog.stream_synchronize(self._handle)

    def _destroy_prog(self):
        """Resolve a Program for resource cleanup.

        Falls back to the current runtime when the owner has been collected, which is safe because CUDAContext is a
        singleton so the CUDA stream handle remains valid.
        """
        prog = self._prog()
        if prog is None:
            try:
                return impl.get_runtime().prog
            except Exception:
                return None
        return prog

    def destroy(self):
        """Explicitly destroy the stream. Safe to call multiple times.

        No-op for streams wrapping external handles (created via Stream(ptr) without a prog_ref).
        """
        if self._handle != 0 and self._prog_ref is not None:
            prog = self._destroy_prog()
            if prog is not None:
                prog.stream_destroy(self._handle)
            self._handle = 0

    def __del__(self):
        if self._handle != 0 and self._prog_ref is not None:
            prog = self._destroy_prog()
            if prog is not None:
                try:
                    prog.stream_destroy(self._handle)
                    self._handle = 0
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.destroy()


class Event:
    """Wraps a backend-specific GPU event for stream synchronization.

    On backends without native events (e.g. CPU), this is a no-op object. Call destroy() explicitly or use as
    a context manager to ensure cleanup.
    """

    def __init__(self, handle: int, prog_ref: weakref.ref | None = None):
        self._handle = handle
        self._prog_ref = prog_ref

    @property
    def handle(self) -> int:
        return self._handle

    def _prog(self):
        """Resolve the owning Program, or None if the owner was collected."""
        if self._prog_ref is not None:
            return self._prog_ref()
        return impl.get_runtime().prog

    def _require_prog(self):
        prog = self._prog()
        if prog is None:
            raise RuntimeError("Event's owning Program has been destroyed (e.g. after qd.reset())")
        return prog

    def record(self, qd_stream: Stream | None = None):
        """Record this event on a stream. None means the default stream."""
        stream_handle = qd_stream.handle if qd_stream is not None else 0
        self._require_prog().event_record(self._handle, stream_handle)

    def wait(self, qd_stream: Stream | None = None):
        """Make a stream wait for this event. None means the default stream."""
        stream_handle = qd_stream.handle if qd_stream is not None else 0
        self._require_prog().stream_wait_event(stream_handle, self._handle)

    def synchronize(self):
        """Block the host until this event has been reached."""
        self._require_prog().event_synchronize(self._handle)

    def _destroy_prog(self):
        """Resolve a Program for resource cleanup.

        Falls back to the current runtime when the owner has been collected, which is safe because CUDAContext is a
        singleton so the CUDA event handle remains valid.
        """
        prog = self._prog()
        if prog is None:
            try:
                return impl.get_runtime().prog
            except Exception:
                return None
        return prog

    def destroy(self):
        """Explicitly destroy the event. Safe to call multiple times.

        No-op for events wrapping external handles (created via Event(ptr) without a prog_ref).
        """
        if self._handle != 0 and self._prog_ref is not None:
            prog = self._destroy_prog()
            if prog is not None:
                prog.event_destroy(self._handle)
            self._handle = 0

    def __del__(self):
        if self._handle != 0 and self._prog_ref is not None:
            prog = self._destroy_prog()
            if prog is not None:
                try:
                    prog.event_destroy(self._handle)
                    self._handle = 0
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.destroy()


def create_stream() -> Stream:
    """Create a new GPU stream for concurrent kernel execution."""
    prog = impl.get_runtime().prog
    handle = prog.stream_create()
    return Stream(handle, _get_prog_weakref())


def create_event() -> Event:
    """Create a new GPU event for stream synchronization."""
    prog = impl.get_runtime().prog
    handle = prog.event_create()
    return Event(handle, _get_prog_weakref())


@contextmanager
def stream_parallel():
    """Run top-level for loops in this block on separate GPU streams.

    Used inside @qd.kernel. At Python runtime (outside kernels), this is a no-op. During kernel compilation, the AST
    transformer calls into the C++ ASTBuilder to tag loops with a stream-parallel group ID.
    """
    yield


__all__ = ["Stream", "Event", "create_stream", "create_event", "stream_parallel"]
