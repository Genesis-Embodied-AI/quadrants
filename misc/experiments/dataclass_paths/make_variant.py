"""Apply only diagnostic-path representation changes to either pinned resolver snapshot.

Usage: python make_variant.py CHECKOUT tuple|list
The checkout must contain the original resolver. Commit the resulting patch before running experiments.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("variant", choices=("tuple", "list", "elided"))
    args = parser.parse_args()
    folder = args.checkout / "python/quadrants/lang"
    file = folder / "_runtime_dataclass.py"
    source = file.read_text()
    kind = args.variant
    if kind == "elided":
        # Attribution-only bound: broken diagnostics, never a correctness or FEM candidate.
        for name in ("_runtime_dataclass.py", "_runtime_dataclass_guards.py"):
            target = folder / name
            if target.exists():
                text = target.read_text().replace('f"{path}.{name}"', "path")
                target.write_text(text.replace('f"{path}.{field.name}"', "path"))
        return
    annotation = "str | tuple[str, ...]" if kind == "tuple" else "str | list[str]"
    source = source.replace("path: str,", f"path: {annotation},")
    conversion = "(path,)" if kind == "tuple" else "[path]"
    # Preserve public root-string calls, including direct calls to infer_member_type.
    for marker in ("    if isinstance(value, Tensor):", '    if not hasattr(type(value), "__dataclass_fields__"):'):
        assert source.count(marker) == 1
        source = source.replace(marker, f"    if isinstance(path, str):\n        path = {conversion}\n" + marker)
    source = source.replace("{path!r}", "{'.'.join(path)!r}")
    if kind == "tuple":
        source = source.replace('f"{path}.{name}"', "path + (name,)")
        source = source.replace('f"{path}.{field.name}"', "path + (field.name,)")
    else:
        old = '        members = {name: infer_member_type(child, f"{path}.{name}", visiting) for name, child in value.items}'
        assert old in source
        source = source.replace(
            old,
            """        members = {}
        for name, child in value.items:
            path.append(name)
            try:
                members[name] = infer_member_type(child, path, visiting)
            finally:
                path.pop()""",
        )
        if "resolved.type = infer_member_type(\n" in source:
            old = """            resolved.type = infer_member_type(
                child, f"{path}.{field.name}", visiting, child_final_parent, validated_keys
            )"""
            new = """            path.append(field.name)
            try:
                resolved.type = infer_member_type(child, path, visiting, child_final_parent, validated_keys)
            finally:
                path.pop()"""
        else:
            old = '            member_type = infer_member_type(child, f"{path}.{name}", visiting)'
            new = """            path.append(name)
            try:
                member_type = infer_member_type(child, path, visiting)
            finally:
                path.pop()"""
        assert old in source
        source = source.replace(old, new)
    file.write_text(source)
    guard = folder / "_runtime_dataclass_guards.py"
    if guard.exists():
        source = guard.read_text()
        if kind == "tuple":
            source = source.replace('f"{path}.{name}"', "path + (name,)")
        else:
            old = """            if infer_member_type(child, f"{path}.{name}", visiting) is not member_type:
                return False"""
            new = """            path.append(name)
            try:
                if infer_member_type(child, path, visiting) is not member_type:
                    return False
            finally:
                path.pop()"""
            assert old in source
            source = source.replace(old, new)
        guard.write_text(source)


if __name__ == "__main__":
    main()
