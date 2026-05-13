# AGENTS.md

Instructions for AI coding agents and AI code-review agents working in this repository.

## Minimize contact area between new and existing code

In order to reduce the risk of changes, changes should minimize the "contact area" between new code and existing code where possible. This applies especially to new and experimental features.

- Ideally, a new feature should be added in new files as much as possible, and modifications to existing code kept to a minimum.
- When there is a choice of two ways to implement a new feature, and one is more integrated and one is more partitioned, choose the more partitioned one - the one with the minimum contact area with existing code.
- When existing code must be touched, keep the diff small and localized.

Code-review agents: flag PRs that unnecessarily expand the contact area with existing code when a more partitioned alternative is feasible.

## Keep user-facing docs in sync with public API changes

Any change that modifies the public API in any way, or changes its usage, should have associated changes or additions to the docs in the `docs/` folder.

- Very low-level things (e.g. tightening up of exception messages and similar) do not need to be documented.
- Documentation should be targeted at end-users of Quadrants, not at Quadrants developers.

Code-review agents: flag PRs that change the public API or its usage without corresponding `docs/` updates.

## Prefer the highest-level language that gives similar performance

Code should be written in the highest-level language possible that will give similar performance to a lower-level alternative.

- Prefer Python to C++.
- Prefer `.cu` to `.ptx`.
- Prefer C/C++ over LLVM IR.
- Prefer C/C++ to LLVM Builder.
- Prefer Python over Bash.

Only drop down to a lower-level language when there is a concrete performance (or other technical) reason that the higher-level option cannot meet.

Code-review agents: flag PRs that use a lower-level language than necessary when a higher-level option would give similar performance.
