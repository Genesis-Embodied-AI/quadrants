# -*- coding: utf-8 -*-

# -- stdlib --
import argparse
import inspect
import os
import sys
from typing import Any, Callable

from .bootstrap import get_cache_home  # noqa: F401 This is a re-export

# -- third party --
# -- own --
from .escapes import escape_codes

# -- code --
options: argparse.Namespace | None = None


def info(msg: str) -> None:
    B = escape_codes["bold_blue"]
    N = escape_codes["reset"]
    print(f"{B}:: INFO {msg}{N}", file=sys.stderr, flush=True)


def warn(msg: str) -> None:
    Y = escape_codes["bold_yellow"]
    N = escape_codes["reset"]
    print(f"{Y}:: WARN {msg}{N}", file=sys.stderr, flush=True)


def error(msg: str) -> None:
    R = escape_codes["bold_red"]
    N = escape_codes["reset"]
    print(f"{R}!! ERROR {msg}{N}", file=sys.stderr, flush=True)
    sys.exit(1)


def banner(msg: str) -> Callable:
    """
    Decorate a function to print a banner before and after it.
    """
    p = lambda s: print(s, file=sys.stderr, flush=True)

    def decorate(f: Callable) -> Callable:
        sig = inspect.signature(f)
        C = escape_codes["bold_cyan"]
        R = escape_codes["bold_red"]
        N = escape_codes["reset"]

        def wrapper(*args, **kwargs):
            _args = sig.bind(*args, **kwargs)
            p(f"{C}:: -----BEGIN {msg}-----{N}".format(**_args.arguments))
            try:
                ret = f(*args, **kwargs)
                p(f"{C}:: -----END {msg}-----{N}".format(**_args.arguments))
                return ret
            except BaseException as e:
                p(f"{R}!! -----EXCEPTION {msg}-----{N}".format(**_args.arguments))
                raise

        return wrapper

    return decorate


def path_prepend(var: str, *paths: Any) -> None:
    """
    Prepend paths to the environment variable.
    """
    value = os.pathsep.join(str(p) for p in paths if p)
    orig = os.environ.get(var, "")
    if orig:
        value += os.pathsep + orig
    os.environ[var] = value
