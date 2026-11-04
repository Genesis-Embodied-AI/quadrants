"""Refine a tuple/list candidate to normalize root strings only at external inference entry points."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("kind", choices=("tuple", "list"))
    args = parser.parse_args()
    target = args.checkout / "python/quadrants/lang/_runtime_dataclass.py"
    source = target.read_text()
    conversion = "(path,)" if args.kind == "tuple" else "[path]"
    annotation = "tuple[str, ...]" if args.kind == "tuple" else "list[str]"
    source = source.replace(f"    if isinstance(path, str):\n        path = {conversion}\n", "")
    source = source.replace(f"str | {annotation}", annotation)
    source = source.replace("infer_member_type(", "_infer_member_type(")
    source = source.replace("visiting, infer_member_type)", "visiting, _infer_member_type)")
    source = source.replace("def resolve_dataclass(", "def _resolve_dataclass(")
    source = source.replace(
        "return resolve_dataclass(value, path, visiting", "return _resolve_dataclass(value, path, visiting"
    )
    parent = "validated_keys" in source
    extra_signature = ", final_parent: str | None = None, validated_keys=None" if parent else ""
    extra_call = ", final_parent, validated_keys" if parent else ""
    result_type = "InferredMemberType | DeferredMemberType" if parent else "Any"
    wrappers = f'''

def infer_member_type(value: Any, path: str, visiting: frozenset[int]{extra_signature}) -> {result_type}:
    """Start a diagnostic path for an external member-inference call."""
    return _infer_member_type(value, {conversion}, visiting{extra_call})


def resolve_dataclass(value: Any, path: str, visiting: frozenset[int] = frozenset(){extra_signature}) -> type:
    """Start an isolated diagnostic path while preserving the root-string calling convention."""
    return _resolve_dataclass(value, {conversion}, visiting{extra_call})
'''
    index = source.index("\ndef _infer_member_type(")
    source = source[:index] + wrappers + source[index:]
    target.write_text(source)


if __name__ == "__main__":
    main()
