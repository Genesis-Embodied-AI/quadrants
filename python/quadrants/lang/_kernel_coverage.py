"""Kernel code coverage via Python AST rewriting.

When enabled (QD_KERNEL_COVERAGE=1), this module rewrites kernel and func ASTs to insert coverage probes - field
stores that record which source lines actually execute on the GPU. At process exit, the collected data is written
to a .coverage file compatible with coverage.py / pytest-cov / diff-cover.

The probes are compiled as ordinary field stores by the existing pipeline, so no C++ changes are needed. When
disabled, this module is never imported and has zero impact on the normal runtime path.
"""

import ast
import atexit
import logging
import os
import threading
import warnings
from typing import TYPE_CHECKING

from coverage import CoverageData  # type: ignore[import-not-found]

import quadrants as qd
from quadrants.lang import impl

if TYPE_CHECKING:
    from quadrants.lang.field import ScalarField

FIELD_VAR_NAME = "_qd_cov"
_MAX_PROBES = int(os.environ.get("QD_COVERAGE_MAX_PROBES", "100000"))

_lock = threading.Lock()
_cov_field: "ScalarField | None" = None
_cov_field_prog: object | None = None  # tracks which Program instance owns _cov_field
_probe_counter: int = 0
# {probe_id: (filepath, absolute_lineno)}
_probe_map: dict[int, tuple[str, int]] = {}
# Accumulated coverage lines surviving across qd.init() resets
_accumulated_lines: dict[str, set[int]] = {}
_reset_hook_installed: bool = False
# Directory for .coverage and _qd_kcov.* files, captured when coverage is first enabled
_coverage_dir: str | None = None


def _harvest_field() -> None:
    """Read probe data from the current field into _accumulated_lines.

    Must be called while the runtime is still alive (before clear()).
    """
    global _cov_field, _cov_field_prog
    with _lock:
        if _cov_field is None or not _probe_map:
            return
        field_ref = _cov_field
        probe_snapshot = dict(_probe_map)
        _cov_field = None
        _cov_field_prog = None
    try:
        arr = field_ref.to_numpy()
    except Exception:
        logging.warning("Failed to read coverage field, coverage data for this session will be lost", exc_info=True)
        return
    with _lock:
        for probe_id, (filepath, lineno) in probe_snapshot.items():
            if probe_id < len(arr) and arr[probe_id] != 0:
                _accumulated_lines.setdefault(filepath, set()).add(lineno)


def _install_reset_hook() -> None:
    """Monkey-patch PyQuadrants.clear() to harvest probes before destruction."""
    global _reset_hook_installed
    if _reset_hook_installed:
        return
    _original_clear = impl.PyQuadrants.clear

    def _hooked_clear(self) -> None:
        _harvest_field()
        _original_clear(self)

    impl.PyQuadrants.clear = _hooked_clear  # type: ignore[assignment]
    _reset_hook_installed = True


def ensure_field_allocated() -> None:
    """Allocate (or re-allocate after qd.init()) the global coverage field."""
    global _cov_field, _cov_field_prog, _coverage_dir
    _install_reset_hook()
    if _coverage_dir is None:
        _coverage_dir = os.getcwd()
    current_prog = impl.get_runtime()._prog
    if _cov_field is not None and _cov_field_prog is current_prog:
        return
    with _lock:
        current_prog = impl.get_runtime()._prog
        if _cov_field is not None and _cov_field_prog is current_prog:
            return
        _cov_field = qd.field(dtype=qd.i32, shape=(_MAX_PROBES,))  # type: ignore[assignment]
        _cov_field_prog = current_prog


def get_field() -> "ScalarField | None":
    with _lock:
        if _cov_field_prog is not impl.get_runtime()._prog:
            return None
        return _cov_field


def rewrite_ast(tree: ast.Module, filepath: str, start_lineno: int) -> ast.Module:
    """Rewrite a kernel/func AST to insert coverage probes.

    Each executable statement at a new source line gets a probe: ``_qd_cov[<probe_id>] = 1``.
    Probes inside if/else bodies only fire when that branch is taken, giving true runtime branch coverage.
    """
    global _probe_counter
    with _lock:
        rewriter = _CoverageASTRewriter(
            field_name=FIELD_VAR_NAME,
            filepath=filepath,
            start_lineno=start_lineno,
            probe_id_start=_probe_counter,
        )
        tree = rewriter.visit(tree)
        ast.fix_missing_locations(tree)
        _probe_counter = rewriter.next_probe_id
        _probe_map.update(rewriter.probe_map)
    return tree


def _detect_arc_mode() -> bool:
    """Detect whether pytest-cov is running in branch (arc) mode.

    Checks _QD_KCOV_ARC env var first (set by the pytest plugin), then falls back to reading .coverage. Defaults to
    False (line mode) when nothing is known, since ``pytest --cov`` without ``--cov-branch`` is the more common
    invocation.
    """
    arc_env = os.environ.get("_QD_KCOV_ARC")
    if arc_env is not None:
        return arc_env == "1"
    try:
        cov_path = os.path.join(_coverage_dir, ".coverage") if _coverage_dir else ".coverage"
        cd = CoverageData(basename=cov_path)
        cd.read()
        if not cd.measured_files():
            return False
        return cd.has_arcs()
    except Exception:
        logging.debug("Failed to detect arc mode from .coverage file, defaulting to line mode", exc_info=True)
        return False


def flush() -> None:
    """Harvest any remaining field data and write all results to a .coverage file.

    If .coverage.kernel already exists (e.g. from a prior test phase), the new data is merged into it so nothing
    is lost across multiple invocations.
    """
    _harvest_field()

    with _lock:
        if not _accumulated_lines:
            return
        snapshot = {f: set(lines) for f, lines in _accumulated_lines.items()}

    base_dir = _coverage_dir or os.getcwd()
    kernel_path = os.path.join(base_dir, f"_qd_kcov.{os.getpid()}")
    use_arcs = _detect_arc_mode()

    cov = CoverageData(basename=kernel_path)
    if use_arcs:
        arcs_by_file: dict[str, list[tuple[int, int]]] = {}
        for filepath, lines in snapshot.items():
            # Emit only entry/exit arcs per line - we know which lines ran but not the actual transitions
            # between them, so we avoid fabricating inter-line arcs that would misrepresent branch coverage.
            arcs = []
            for line in sorted(lines):
                arcs.append((-1, line))
                arcs.append((line, -1))
            arcs_by_file[filepath] = arcs
        cov.add_arcs(arcs_by_file)
    else:
        cov.add_lines({f: sorted(lines) for f, lines in snapshot.items()})
    cov.write()


_capacity_warning_emitted = False


class _CoverageASTRewriter(ast.NodeTransformer):
    """Insert coverage probes before each statement at a new source line."""

    def __init__(self, field_name: str, filepath: str, start_lineno: int, probe_id_start: int) -> None:
        self._field_name = field_name
        self._filepath = filepath
        self._start_lineno = start_lineno
        self.next_probe_id = probe_id_start
        self._seen_lines: set[int] = set()
        self.probe_map: dict[int, tuple[str, int]] = {}

    def _make_probe(self, abs_lineno: int, rel_lineno: int, col_offset: int) -> ast.Assign | None:
        global _capacity_warning_emitted
        probe_id = self.next_probe_id
        if probe_id >= _MAX_PROBES:
            if not _capacity_warning_emitted:
                warnings.warn(
                    f"Kernel coverage probe capacity ({_MAX_PROBES}) exceeded. "
                    f"Additional kernel lines will not be tracked. "
                    f"Set QD_COVERAGE_MAX_PROBES to a higher value.",
                    stacklevel=2,
                )
                _capacity_warning_emitted = True
            return None
        self.probe_map[probe_id] = (self._filepath, abs_lineno)
        self.next_probe_id += 1
        node = ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id=self._field_name, ctx=ast.Load()),
                    slice=ast.Constant(value=probe_id),
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=1),
            lineno=rel_lineno,
            col_offset=col_offset,
            end_lineno=rel_lineno,
            end_col_offset=col_offset,
        )
        return node

    def _instrument_body(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for stmt in stmts:
            rel_lineno = getattr(stmt, "lineno", None)
            if rel_lineno is not None:
                abs_lineno = rel_lineno + self._start_lineno - 1
                if abs_lineno not in self._seen_lines:
                    self._seen_lines.add(abs_lineno)
                    col = getattr(stmt, "col_offset", 0)
                    probe = self._make_probe(abs_lineno, rel_lineno, col)
                    if probe is not None:
                        result.append(probe)
            result.append(self.visit(stmt))
        return result

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.body = self._instrument_body(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.body = self._instrument_body(node.body)
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        node.body = self._instrument_body(node.body)
        if node.orelse:
            node.orelse = self._instrument_body(node.orelse)
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        node.body = self._instrument_body(node.body)
        if node.orelse:
            node.orelse = self._instrument_body(node.orelse)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        node.body = self._instrument_body(node.body)
        if node.orelse:
            node.orelse = self._instrument_body(node.orelse)
        return node

    def visit_With(self, node: ast.With) -> ast.With:
        node.body = self._instrument_body(node.body)
        return node

    def visit_Try(self, node: ast.Try) -> ast.Try:
        node.body = self._instrument_body(node.body)
        for handler in node.handlers:
            handler.body = self._instrument_body(handler.body)
        if node.orelse:
            node.orelse = self._instrument_body(node.orelse)
        if node.finalbody:
            node.finalbody = self._instrument_body(node.finalbody)
        return node


atexit.register(flush)
