"""Allocation-time field initialization, replayed after placement is complete."""

import numbers


def prepare_fill(value, element_shape=()):
    """Validate and snapshot an initializer without allocating or materializing any fields."""
    if value is None or isinstance(value, numbers.Number):
        return value
    # Import lazily: matrix.py also uses this module when constructing fields.
    from quadrants.lang.matrix import Matrix  # pylint: disable=C0415

    if isinstance(value, Matrix):
        value = value.to_list()
    if not element_shape or not isinstance(value, (list, tuple)) or len(value) != element_shape[0]:
        raise ValueError(f"fill must be a number or a sequence with element shape {element_shape}")
    result = tuple(prepare_fill(item, element_shape[1:]) for item in value)
    if any(item is None for item in result):
        raise ValueError("fill components must be numbers")
    if len(element_shape) > 1 and any(not isinstance(item, tuple) for item in result):
        raise ValueError(f"fill must have element shape {element_shape}")
    return result


class FieldInitializers:
    """Runtime-owned initializers; ordinary field.fill calls never enter this queue."""

    def __init__(self):
        self.pending = []

    def add(self, field, value):
        if value is not None:
            self.pending.append((field, value))

    def clear(self):
        self.pending.clear()

    def run(self):
        if not self.pending:
            return
        # Fill kernels materialize recursively. Detach the queue before launching any of them.
        pending, self.pending = self.pending, []
        for field, value in pending:
            field.fill(value)
