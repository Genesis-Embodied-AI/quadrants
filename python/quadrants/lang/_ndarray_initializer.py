"""Allocation-time initialization for ndarray factories."""


def initialize_ndarray(array, value):
    """Initialize an allocated array, including visibility to zero-copy consumers."""
    if value is None:
        return array

    from quadrants.lang import impl  # pylint: disable=import-outside-toplevel
    from quadrants.lang._field_initializer import (  # pylint: disable=import-outside-toplevel
        capture_fill_value,
    )

    if impl.is_python_backend():
        captured_fill_value = capture_fill_value(value, tuple(array.size()[len(array.shape) :]))
        array.copy_(array.new_tensor(captured_fill_value))
        return array
    captured_fill_value = capture_fill_value(value, array.element_shape)
    array.fill(captured_fill_value)
    # A consumer of to_dlpack() may use a different device queue. Complete allocation-time initialization before
    # returning the tensor so it is safe to export without an intervening explicit sync.
    impl.get_runtime().prog.synchronize()
    return array
