import hashlib
from typing import Iterable


def hash_iterable_strings(strings: Iterable[str], separator: str = "_") -> str:
    h = hashlib.sha256()
    separator_enc = separator.encode("utf-8")
    for v in strings:
        h.update(v.encode("utf-8"))
        h.update(separator_enc)
    return h.hexdigest()
