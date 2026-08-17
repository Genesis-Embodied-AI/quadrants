# Self-documenting `qd.init` / `CompileConfig` options (Route B1)

Status: spike / design. Author: config-self-documenting-b1 branch.

## Problem

Today a `qd.init(...)` option's information is scattered across at least three
places that can (and do) drift:

1. `quadrants/program/compile_config.h` - field name + type + (some) defaults
   via in-class initializers.
2. `quadrants/program/compile_config.cpp` - more defaults, set in the ctor.
3. `python/quadrants/lang/misc.py` - frontend overrides (e.g. `offline_cache`
   is `false` in the struct but forced `True` at `qd.init` time).
4. `quadrants/python/export_lang.cpp` - a hand-maintained `.def_rw(...)` line
   per field for the nanobind binding.
5. `docs/source/user_guide/init_options.md` - hand-written prose describing a
   subset of options, with defaults copied by hand.

The doc review that started this work found a concrete drift bug: the doc cited
`compile_config.h` as the "source of truth" and stated `offline_cache` defaults
to `True`, but the struct initializes it to `false` (the `True` comes from the
Python frontend). Copies drift; that is the core problem.

## Goal

One source of truth per option, where the **doc string and the default value
live in the same place**, authored in Python, and everything else (the C++
struct fields, the C++ defaults, the nanobind bindings, and the user-guide
table) is generated from it. "Self-documenting" = the doc is at the definition.

## Chosen approach: B1 (Python schema -> codegen)

A pure-data Python module (`tools/config_codegen/schema.py`) is the single
source of truth. Each option records `name`, `cpp_type`, `py_type`, `default`,
and `doc` together. From it we generate:

- C++ struct member declarations with in-class initializers + doc comments
  (`compile_config.fields.generated.inc`).
- A ctor fragment for computed (non-literal) defaults
  (`compile_config.ctor.generated.inc`).
- The nanobind `.def_rw(...)` block (`compile_config.bindings.generated.inc`).
- The user-guide reference table (via a Sphinx directive that reads the same
  schema, so docs cannot drift from code).

The schema imports nothing from `quadrants`, so it is safe to import at C++
build time (before the extension exists) and at docs build time.

### Why B1 over the alternatives

- Doxygen+Breathe (Route A): co-locates doc with code but forces prose into C++
  headers and does not solve defaults living in two C++ spots; also pulls
  Doxygen into the docs build.
- Python schema + CI parity test (Route B2): keeps a hand-written C++ struct in
  sync via a test. Lower effort but the default is still duplicated (schema and
  struct), which is exactly the drift we are removing.

B1 is the only option that puts doc+default in one Python location with no
duplication.

## Schema format

```python
@dataclass(frozen=True)
class Option:
    name: str          # field / kwarg / QD_<UPPER> env var name
    cpp_type: str      # "bool" | "int" | "std::size_t" | "uint64_t" | "std::string" | ...
    py_type: str       # "bool" | "int" | "str" (what the user passes)
    default: Any       # a literal, or Computed(cpp_expr=..., doc=...)
    doc: str           # the single end-user description
```

Computed defaults (values not known at compile time, e.g.
`offline_cache_file_path = get_repo_dir() + "qdcache"`, `cpu_max_num_threads`,
arch autodetection) use a `Computed` marker: the field is declared without an
in-class initializer and assigned in the generated ctor fragment, exactly as
the current code does by hand.

## C++ integration

`compile_config.h` includes the generated fragment inside the struct body:

```cpp
struct CompileConfig {
#include "quadrants/program/compile_config.fields.generated.inc"
  // (any remaining hand-written members during migration)
  CompileConfig();
  void fit();
};
```

`compile_config.cpp` includes the ctor fragment for computed defaults;
`export_lang.cpp` includes the bindings fragment inside
`nb::class_<CompileConfig>(...)` ... the trailing `;` stays in the hand-written
file so the generated block is a pure chain of `.def_rw(...)` calls.

## Build integration

Mirror the existing generated-header precedent in this repo:
`CMakeLists.txt:236` already does
`configure_file(quadrants/common/version.h.in ... version.h)`, writing a
generated header into the source tree at configure time. We add an equivalent
step that runs the codegen before the C++ compile:

```cmake
add_custom_command(
  OUTPUT ${CMAKE_SOURCE_DIR}/quadrants/program/compile_config.fields.generated.inc
         ${CMAKE_SOURCE_DIR}/quadrants/program/compile_config.ctor.generated.inc
         ${CMAKE_SOURCE_DIR}/quadrants/python/compile_config.bindings.generated.inc
  COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/config_codegen/generate.py
          --out-dir ${CMAKE_SOURCE_DIR}
  DEPENDS ${CMAKE_SOURCE_DIR}/tools/config_codegen/schema.py
          ${CMAKE_SOURCE_DIR}/tools/config_codegen/generate.py
  COMMENT "Generating CompileConfig from tools/config_codegen/schema.py")
add_custom_target(config_codegen DEPENDS <the .inc outputs>)
# core lib depends on config_codegen so generation runs first.
```

The `quadrants/program/*` glob at `cmake/QuadrantsCore.cmake:63-68` already
sweeps the program dir, so no source-list edits are needed for `.inc` includes.
Generated `*.generated.inc` files are added to `.gitignore` (like `version.h`).

## Docs integration

Add `tools/config_codegen` to `sys.path` in `docs/source/conf.py` and register
the `qd-config-options` directive (see `sphinx_ext.py`). `init_options.md` keeps
its hand-written prose for the commonly-tuned knobs but replaces the manual
"full reference" list with a single directive call:

```markdown
## Full option reference

```{qd-config-options}
```
```

Because the committed markdown contains only the directive (not a hand-copied
table), the doc-quality checker sees no drift-prone content, and the rendered
table's defaults/types/docs come straight from the schema.

## Edge cases

- `offline_cache`: canonical default becomes `True` in the schema; the frontend
  override in `misc.py:388` is removed (verified equivalent: struct `false` +
  forced `True` == default `True`). This alone fixes the drift bug that started
  this work.
- Computed defaults: handled via `Computed` (ctor assignment), documented with a
  human-readable string (e.g. "<cache dir>/qdcache").
- `fit()`-time interactions (e.g. `debug` implies `check_out_of_bound`) are
  runtime behavior, not defaults, and stay in `compile_config.cpp`. The schema
  documents the plain default; prose covers interactions.
- Non-user-facing / internal fields: migrate them too (so the struct is fully
  generated) but mark `user_facing=False` (added in phase 1) so the docs
  directive can filter them out.

## Migration plan (phased, each phase independently reviewable)

- Phase 0 (this spike): schema + generator + Sphinx directive for the
  commonly-tuned subset; show generated C++ and rendered table. No build wiring
  yet; the real struct is untouched.
- Phase 1: extend the schema to ALL current `CompileConfig` fields so the
  generator emits the struct/bindings verbatim (byte-for-byte equivalent
  defaults). Add a parity check that compares generated defaults against the
  current struct before switching. Then include the `.inc` files and delete the
  hand-written field/binding lines. Wire CMake. Build on CI + cluster; run the
  test suite.
- Phase 2: point `init_options.md` full-reference at the directive; drop the
  hand-copied list.
- Phase 3: remove the `misc.py` `offline_cache` override; add `user_facing`
  filtering.

## Risks

- A partial/incorrect schema could change a default and silently alter behavior.
  Mitigated by the phase-1 parity check and full test run before deleting any
  hand-written code.
- Generating into the source tree can surprise contributors. Mitigated by the
  existing `version.h` precedent, `.gitignore`, and a clear "DO NOT EDIT" banner
  in every generated file.

## Non-goals

- Changing option semantics or runtime behavior.
- Generating the environment-variable parser (it already derives `QD_<UPPER>`
  names generically from the field list).
