# Compiler internals: Visitors in Quadrants

Quadrants uses visitors to implement compiler passes over its intermediate representation (IR). For an introduction to the design pattern, see [Visitor pattern on Wikipedia](https://en.wikipedia.org/wiki/Visitor_pattern). This page explains the implementation and conventions used in Quadrants.

## Where to look

| Source | Role |
|---|---|
| [`ir.h`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/ir/ir.h) | Defines `IRVisitor` and the `QD_DEFINE_ACCEPT` macro. |
| [`statements.inc.h`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/inc/statements.inc.h) | Registers statement types used to generate the visitor overloads; includes the frontend statement list. |
| [`visitors.h`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/ir/visitors.h) | Declares `BasicStmtVisitor` and its container traversal hooks. |
| [`basic_stmt_visitor.cpp`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/ir/basic_stmt_visitor.cpp) | Implements traversal through blocks, branches, loops, and offloaded tasks. |
| [`remove_loop_unique.cpp`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/transforms/remove_loop_unique.cpp) | A small example of a transformation implemented as a visitor. |

## How `accept` selects a `visit` overload

`IRVisitor` declares virtual `visit` overloads for `Block` and the registered statement types. Statement classes use `QD_DEFINE_ACCEPT`, often through `QD_DEFINE_ACCEPT_AND_CLONE`, to supply this implementation:

```cpp
void accept(IRVisitor *visitor) override {
  visitor->visit(this);
}
```

For example, when a `Stmt *` points to a `LoopUniqueStmt`, calling `stmt->accept(&pass)` works in three steps:

1. Virtual dispatch selects the `LoopUniqueStmt` implementation of `accept`.
2. Inside that implementation, `this` has static type `LoopUniqueStmt *`, so overload resolution selects `visit(LoopUniqueStmt *)`.
3. Virtual dispatch calls the concrete pass's override of that overload.

Calling `visit` directly through an `IRVisitor *` with a `Stmt *` argument instead selects `visit(Stmt *)`. C++ overload resolution uses the argument's static type; it does not discover the concrete statement type at runtime. Use `accept` when dispatching from a base-class pointer.

## Traversal and unhandled statements

`IRVisitor` does not provide recursive traversal. By default, its unimplemented overloads report an error. Its `allow_undefined_visitor` flag permits unhandled types; with that enabled, `invoke_default_visitor` additionally forwards unhandled typed visits to `visit(Stmt *)`.

`BasicStmtVisitor` enables `allow_undefined_visitor` and leaves `invoke_default_visitor` disabled. It therefore ignores unhandled leaf statements while providing traversal for the containers implemented in `basic_stmt_visitor.cpp`:

- `visit(Block *)` calls `accept(this)` on each statement in the block.
- Branch and loop visitors visit their child blocks.
- `visit(OffloadedStmt *)` uses `all_blocks_accept(this)` to visit the task's blocks.

These container visitors call `preprocess_container_stmt(Stmt *)` before descending. Override that hook when you need to inspect containers while retaining the existing traversal.

If you override a container's `visit`, its inherited implementation does not run automatically. Call the corresponding `BasicStmtVisitor::visit(stmt)` to retain its traversal, or explicitly visit the child blocks you need. For an example of deliberate custom traversal, see [`ExtractConstant::visit(OffloadedStmt *)`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/transforms/extract_constant.cpp), which visits the task body while changing the destination block for extracted constants.

## Example: removing `LoopUniqueStmt`

The `RemoveLoopUnique` pass derives from `BasicStmtVisitor`, inherits its traversal, and overrides one statement type. Its essential code is:

```cpp
class RemoveLoopUnique : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;

  // Redirect uses to the input and schedule this statement for removal.
  void visit(LoopUniqueStmt *stmt) override {
    stmt->replace_usages_with(stmt->input);
    modifier.erase(stmt);
  }

  // Traverse the IR, then apply the queued removals.
  static bool run(IRNode *node) {
    RemoveLoopUnique pass;
    node->accept(&pass);
    return pass.modifier.modify_ir();
  }
};
```

`using BasicStmtVisitor::visit` keeps the inherited overloads visible for direct calls on the derived visitor. The typed override handles `LoopUniqueStmt`; the inherited container visitors reach those statements inside nested blocks.

`DelayedIRModifier` queues removals until traversal finishes. Although `BasicStmtVisitor::visit(Block *)` copies the block's statement pointers before iterating, that copy does not keep erased statements alive. Follow the pass's deferred-removal approach when deleting statements during a traversal.

## Adding a visitor or statement type

For a pass that handles selected statement types and otherwise walks nested blocks, derive from `BasicStmtVisitor`, expose its overloads with `using BasicStmtVisitor::visit`, and override the types you need. Start traversal with `root->accept(&pass)`. Choose `IRVisitor` instead when you want to define traversal yourself and have unhandled types report errors by default.

When adding a statement type, register it in `statements.inc.h` or its included frontend list and provide its `accept` implementation, following neighboring statement classes. Review the passes that must understand the new type: registration creates an overload, but does not implement its semantics. If the statement contains child blocks, also decide how `BasicStmtVisitor` should traverse them. Its permissive handling of unknown types can otherwise leave those children unvisited.
