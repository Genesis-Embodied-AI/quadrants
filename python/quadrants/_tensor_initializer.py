"""Initialization for the ndarray branch of the tensor factories."""


def initialize_ndarray(array, value):
    """Initialize an allocated array, including visibility to zero-copy consumers."""
    if value is None:
        return array

    from quadrants.lang import impl  # pylint: disable=import-outside-toplevel
    from quadrants.lang._field_initializer import (  # pylint: disable=import-outside-toplevel
        prepare_fill,
    )

    value = prepare_fill(value, array.element_shape)
    array.fill(value)
    # A consumer of to_dlpack() may use a different device queue. Complete allocation-time initialization before
    # returning the tensor so it is safe to export without an intervening explicit sync.
    impl.get_runtime().prog.synchronize()
    return array
