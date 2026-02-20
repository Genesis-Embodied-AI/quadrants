# type: ignore
from quadrants._lib import core as _qd_core

__version__ = (
    _qd_core.get_version_major(),
    _qd_core.get_version_minor(),
    _qd_core.get_version_patch(),
)
__version_str__ = ".".join(map(str, __version__))

from quadrants import (
    ad,
    algorithms,
    experimental,
    linalg,
    math,
    sparse,
    tools,
    types,
)
from quadrants._funcs import *
from quadrants._lib.utils import warn_restricted_version
from quadrants._logging import *
from quadrants._snode import *
from quadrants.lang import *  # pylint: disable=W0622 # TODO(archibate): It's `quadrants.lang.core` overriding `quadrants.core`
from quadrants.lang.intrinsics import *
from quadrants.types.annotations import *

# Provide a shortcut to types since they're commonly used.
from quadrants.types.primitive_types import *


def __getattr__(attr):
    if attr == "cfg":
        return None if lang.impl.get_runtime()._prog is None else lang.impl.current_cfg()
    raise AttributeError(f"module '{__name__}' has no attribute '{attr}'")


warn_restricted_version()
del warn_restricted_version

__all__ = [
    "ad",
    "algorithms",
    "experimental",
    "linalg",
    "math",
    "sparse",
    "tools",
    "types",
]
