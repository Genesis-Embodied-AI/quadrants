from quadrants.lang import impl


class Stream:
    """Wraps a backend-specific GPU stream for concurrent kernel execution.

    On backends without native streams (e.g. CPU), this is a no-op object.
    """

    def __init__(self, handle: int):
        self._handle = handle

    @property
    def handle(self) -> int:
        return self._handle

    def synchronize(self):
        """Block until all operations on this stream complete."""
        prog = impl.get_runtime().prog
        prog.stream_synchronize(self._handle)

    def destroy(self):
        """Explicitly destroy the stream. Safe to call multiple times."""
        if self._handle != 0:
            prog = impl.get_runtime().prog
            prog.stream_destroy(self._handle)
            self._handle = 0

    def __del__(self):
        if self._handle != 0:
            try:
                self.destroy()
            except Exception:
                pass


class Event:
    """Wraps a backend-specific GPU event for stream synchronization.

    On backends without native events (e.g. CPU), this is a no-op object.
    """

    def __init__(self, handle: int):
        self._handle = handle

    @property
    def handle(self) -> int:
        return self._handle

    def record(self, stream: Stream | None = None):
        """Record this event on a stream. None means the default stream."""
        prog = impl.get_runtime().prog
        stream_handle = stream.handle if stream is not None else 0
        prog.event_record(self._handle, stream_handle)

    def wait(self, qd_stream: Stream | None = None):
        """Make a stream wait for this event. None means the default stream."""
        prog = impl.get_runtime().prog
        stream_handle = qd_stream.handle if qd_stream is not None else 0
        prog.stream_wait_event(stream_handle, self._handle)

    def synchronize(self):
        """Block the host until this event has been reached."""
        prog = impl.get_runtime().prog
        prog.event_synchronize(self._handle)

    def destroy(self):
        """Explicitly destroy the event. Safe to call multiple times."""
        if self._handle != 0:
            prog = impl.get_runtime().prog
            prog.event_destroy(self._handle)
            self._handle = 0

    def __del__(self):
        if self._handle != 0:
            try:
                self.destroy()
            except Exception:
                pass


def create_stream() -> Stream:
    """Create a new GPU stream for concurrent kernel execution."""
    prog = impl.get_runtime().prog
    handle = prog.stream_create()
    return Stream(handle)


def create_event() -> Event:
    """Create a new GPU event for stream synchronization."""
    prog = impl.get_runtime().prog
    handle = prog.event_create()
    return Event(handle)


__all__ = ["Stream", "Event", "create_stream", "create_event"]
