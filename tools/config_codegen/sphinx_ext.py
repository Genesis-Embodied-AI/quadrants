"""Sphinx extension: the ``qd-config-options`` directive.

Renders the option schema (``schema.py``) as a table of
name / type / default / description. Because it reads the SAME schema that
generates the C++ struct and defaults, the rendered docs can never drift from
the code.

To enable, add the directory containing this file to ``sys.path`` in
``docs/source/conf.py`` and append ``"sphinx_ext"`` to ``extensions``. Then in
any markdown/rst page::

    ```{qd-config-options}
    ```

The heavy lifting (turning the schema into rows) lives in ``generate.to_rows``
so the directive and the markdown fallback share one code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import to_rows  # noqa: E402

try:  # docutils is only importable inside a Sphinx/docutils environment.
    from docutils import nodes
    from docutils.parsers.rst import Directive
except ImportError:  # pragma: no cover - allows importing this module standalone
    nodes = None
    Directive = object


HEADERS = ("Option", "Type", "Default", "Description")


def _build_table(rows):
    table = nodes.table()
    tgroup = nodes.tgroup(cols=len(HEADERS))
    table += tgroup
    for _ in HEADERS:
        tgroup += nodes.colspec(colwidth=1)

    thead = nodes.thead()
    tgroup += thead
    header_row = nodes.row()
    thead += header_row
    for label in HEADERS:
        entry = nodes.entry()
        entry += nodes.paragraph(text=label)
        header_row += entry

    tbody = nodes.tbody()
    tgroup += tbody
    for name, py_type, default, doc in rows:
        row = nodes.row()
        tbody += row
        for cell, literal in ((name, True), (py_type, True), (default, True), (doc, False)):
            entry = nodes.entry()
            if literal:
                para = nodes.paragraph()
                para += nodes.literal(text=str(cell))
            else:
                para = nodes.paragraph(text=str(cell))
            entry += para
            row += entry
    return table


class QdConfigOptions(Directive):
    """Render the qd.init option schema as a table."""

    has_content = False

    def run(self):
        return [_build_table(to_rows())]


def setup(app):
    app.add_directive("qd-config-options", QdConfigOptions)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
