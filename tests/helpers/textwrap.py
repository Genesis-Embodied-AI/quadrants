def dedent(indent_size: int, v: str) -> str:
    lines = []
    for line in v.split("\n"):
        line = line[indent_size:]
        lines.append(line)
    return "\n".join(lines)
