"""Track Python reads of the random seed for the life of the process."""

from quadrants._lib.core.quadrants_python import CompileConfig

random_seed_read = False

_seed_property = vars(CompileConfig)["random_seed"]


def _get_random_seed(config):
    global random_seed_read  # pylint: disable=global-statement
    random_seed_read = True
    return _seed_property.__get__(config)


setattr(CompileConfig, "random_seed", _seed_property.getter(_get_random_seed))


def random_seed_for_cache_key(config):
    """Read the seed for a cache key without recording a Python program dependency."""
    return _seed_property.__get__(config)
