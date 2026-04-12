# type: ignore

from quadrants.lang.simt import block, grid, subgroup, warp

__all__ = [
    "warp",
    "subgroup",
    "block",
    "grid",
    "min_subgroup_size",
    "max_subgroup_size",
]


def min_subgroup_size():
    """Return the minimum subgroup size supported by the current device."""
    from quadrants.lang.impl import (  # pylint: disable=import-outside-toplevel
        get_runtime,
    )

    return get_runtime().prog.get_min_subgroup_size()


def max_subgroup_size():
    """Return the maximum subgroup size supported by the current device."""
    from quadrants.lang.impl import (  # pylint: disable=import-outside-toplevel
        get_runtime,
    )

    return get_runtime().prog.get_max_subgroup_size()
