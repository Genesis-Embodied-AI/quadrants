# Compiler internals: Visitors in Quadrants

Quadrants uses visitors to implement compiler passes over its intermediate representation (IR). For the general design pattern, see [Visitor pattern on Wikipedia](https://en.wikipedia.org/wiki/Visitor_pattern).

Start with these implementation files:

- [`IRVisitor` and `QD_DEFINE_ACCEPT`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/ir/ir.h) define the typed `visit` overloads and the `accept` method that dispatches to them.
- [`BasicStmtVisitor`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/ir/visitors.h) provides traversal through nested blocks and containers. Its implementation is in [`basic_stmt_visitor.cpp`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/ir/basic_stmt_visitor.cpp).
- [`RemoveLoopUnique`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/transforms/remove_loop_unique.cpp) is a small example pass: it inherits traversal, handles one statement type, and applies queued removals after visiting the IR.
