from quadrants.lang import impl

from .hash_utils import hash_iterable_strings

EXCLUDE_PREFIXES = ["_", "offline_cache", "print_", "verbose_"]


def hash_compile_config() -> str:
    """
    Calculates a hash string for the current compiler config.

    If any value in the compiler config changes, the hash string changes too.

    Though arguably we might want to blacklist certain keys, such as print_ir_debug,
    which do not affect the compiled kernels, just stuff that gets printed during
    the compilation process.
    """
    config = impl.get_runtime().prog.config()
    config_l = []
    for k in dir(config):
        skip = False
        for prefix in EXCLUDE_PREFIXES:
            if k.startswith(prefix) or k in [""]:
                skip = True
        if skip:
            continue
        v = getattr(config, k)
        config_l.append(f"{k}={v}")
    config_hash = hash_iterable_strings(config_l, separator="\n")
    return config_hash
