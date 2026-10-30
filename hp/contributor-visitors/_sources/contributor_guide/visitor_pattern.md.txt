# Compiler internals: The visitor pattern

## What is the visitor pattern?

The **Visitor Pattern** is a software design pattern that lets you **add new operations to a collection of different object types without modifying those objects**.

It's particularly useful for things like compilers, ASTs (abstract syntax trees), and physics engines.

## 1. The basic problem

Imagine you have several types of expressions:

```python
class Add:
    def __init__(self, left, right):
        self.left = left
        self.right = right

class Multiply:
    def __init__(self, left, right):
        self.left = left
        self.right = right

class Number:
    def __init__(self, value):
        self.value = value
```

You want to perform different operations on these expressions:

- Evaluate them.
- Print them.
- Compile them to machine code.
- Count the number of operations.

One option is to put every operation directly into each class:

```python
class Add:
    def evaluate(self): ...
    def print(self): ...
    def compile(self): ...
    def count_ops(self): ...
```

But as you add new operations, every expression class grows.

**The Visitor Pattern solves this by separating the operations from the objects.**

## 2. How it works

Each object implements a method called `accept`, which receives a visitor.

```python
class Add:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def accept(self, visitor):
        return visitor.visit_add(self)


class Number:
    def __init__(self, value):
        self.value = value

    def accept(self, visitor):
        return visitor.visit_number(self)
```

Now define a visitor that evaluates expressions:

```python
class EvaluateVisitor:
    def visit_add(self, node):
        return (
            node.left.accept(self)
            + node.right.accept(self)
        )

    def visit_number(self, node):
        return node.value
```

And another visitor that prints expressions:

```python
class PrintVisitor:
    def visit_add(self, node):
        return (
            f"({node.left.accept(self)}"
            f" + {node.right.accept(self)})"
        )

    def visit_number(self, node):
        return str(node.value)
```

Use them:

```python
expr = Add(Number(2), Number(3))

print(expr.accept(EvaluateVisitor()))  # 5
print(expr.accept(PrintVisitor()))     # (2 + 3)
```

Notice that **we added printing and evaluation without changing `Add` or `Number`.**

## 3. Why is it called a "visitor"?

Think of each expression as a place that a visitor visits.

```{raw} html
<figure style="max-width: 360px;">
  <svg viewBox="0 0 340 215" width="360" style="max-width: 100%; height: auto;" xmlns="http://www.w3.org/2000/svg">
    <rect x="103" y="8" width="134" height="44" rx="9" fill="#64748b" fill-opacity=".15" stroke="#64748b"/>
    <text x="170" y="26" font-size="12" text-anchor="middle" fill="currentColor" font-weight="600">Add</text>
    <text x="170" y="42" font-size="9" text-anchor="middle" fill="currentColor">accept(visitor)</text>
    <path d="M145 52 L83 89 M195 52 L257 89" stroke="#94a3b8" stroke-width="1.6"/>
    <rect x="18" y="90" width="130" height="42" rx="8" fill="#64748b" fill-opacity=".1" stroke="#94a3b8"/>
    <rect x="192" y="90" width="130" height="42" rx="8" fill="#64748b" fill-opacity=".1" stroke="#94a3b8"/>
    <text x="83" y="107" text-anchor="middle" font-size="12" fill="currentColor">Number</text>
    <text x="83" y="122" text-anchor="middle" font-size="10" fill="currentColor">2</text>
    <text x="257" y="107" text-anchor="middle" font-size="12" fill="currentColor">Number</text>
    <text x="257" y="122" text-anchor="middle" font-size="10" fill="currentColor">3</text>
    <rect x="7" y="165" width="156" height="39" rx="8" fill="#0d9488" fill-opacity=".12" stroke="#0d9488"/>
    <rect x="177" y="165" width="156" height="39" rx="8" fill="#2563eb" fill-opacity=".12" stroke="#2563eb"/>
    <text x="85" y="182" text-anchor="middle" fill="currentColor" font-size="11" font-weight="600">EvaluateVisitor</text>
    <text x="85" y="196" text-anchor="middle" fill="currentColor" font-size="10">Result: 5</text>
    <text x="255" y="182" text-anchor="middle" fill="currentColor" font-size="11" font-weight="600">PrintVisitor</text>
    <text x="255" y="196" text-anchor="middle" fill="currentColor" font-size="10">Result: (2 + 3)</text>
    <path d="M85 165 L85 144 L170 144 L170 140" fill="none" stroke="#0d9488" stroke-width="1.4" stroke-dasharray="4 3"/>
    <path d="M255 165 L255 144 L170 144" fill="none" stroke="#2563eb" stroke-width="1.4" stroke-dasharray="4 3"/>
  </svg>
  <figcaption style="text-align: center;">The same expression tree can be visited by different operations.</figcaption>
</figure>
```

The clever part is this:

```python
def accept(self, visitor):
    return visitor.visit_add(self)
```

The `Add` object knows its own type, so it calls the appropriate method on the visitor.

This is an example of **double dispatch**:

1. `expr.accept(visitor)` selects the implementation based on the expression's type.
2. That implementation calls the corresponding method on the visitor.

The resulting behavior depends on both the expression type and the visitor type.

## 4. When is it useful?

| Situation | Visitor pattern |
|---|---|
| Many node types, frequently adding new operations | Excellent |
| Many operations, frequently adding new node types | Less suitable |
| Compiler AST with multiple passes | Excellent |
| Simple hierarchy with one or two operations | Often unnecessary |

The main drawback is that **adding a new node type requires updating every visitor**.

For example, adding `Divide` means every visitor needs a `visit_divide` implementation (or a fallback).

## 5. The key intuition

The Visitor Pattern essentially converts:

```python
node.evaluate()
node.print()
node.compile()
```

Into:

```python
EvaluateVisitor().visit_add(node)
PrintVisitor().visit_add(node)
CompileVisitor().visit_add(node)
```

With `node.accept(visitor)` handling the dispatch.

**In one sentence:** Visitor lets you organize your code around *operations* rather than around *data types*.

For a compiler like Quadrants, this is particularly relevant: you might have a fixed set of IR node types, but many compiler passes (type checking, optimization, code generation, validation). Each pass can be a separate visitor operating on the same IR.

---

## In Quadrants, the method is always called `visit`, but it is overloaded, right?

Yes. In Quadrants, `visit` is overloaded for different node types. **The argument's static type determines which overload is selected**, and the overloads are virtual so that the concrete visitor supplies the implementation.

For example, in C++:

```cpp
class Visitor {
public:
    virtual ~Visitor() = default;
    virtual void visit(Add* node) = 0;
    virtual void visit(Multiply* node) = 0;
    virtual void visit(Number* node) = 0;
};
```

Each node implements:

```cpp
class Add {
public:
    void accept(Visitor& v) {
        v.visit(this);  // Calls visit(Add*)
    }
};
```

This is the classic Visitor Pattern using **double dispatch**:

1. `node->accept(visitor)` -- virtual dispatch selects the concrete node's `accept`.
2. Inside `accept`, overload resolution selects the correct `visit` based on the static type of `this`; virtual dispatch then invokes that overload on the concrete visitor.

One important subtlety: **C++ overload resolution happens at compile time**, not runtime. It's the virtual `accept` dispatch that provides the runtime selection of node type.

In Quadrants, [`IRVisitor`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/ir/ir.h) declares virtual `visit` overloads, and `QD_DEFINE_ACCEPT` implements `accept(IRVisitor *visitor)` by calling `visitor->visit(this)`.

---

## Does that mean we could skip going through `accept`?

Yes! **You can skip `accept()` if you already know the concrete type of the node at compile time.**

For example:

```cpp
Add* node = ...;
visitor.visit(node);  // Works!
```

But consider:

```cpp
Node* node = new Add(...);
visitor.visit(node);  // Problem!
```

Suppose the visitor has these overloads:

```cpp
void visit(Add*);
void visit(Multiply*);
void visit(Node*);
```

C++ chooses `visit(Node*)`, **not `visit(Add*)`**, even though the object is actually an `Add`.

That's because **overload resolution uses the static type**, not the runtime type.

### Why `accept()` fixes this

```cpp
class Node {
public:
    virtual ~Node() = default;
    virtual void accept(Visitor& v) = 0;
};

class Add : public Node {
public:
    void accept(Visitor& v) override {
        v.visit(this);
    }
};
```

Now:

```cpp
Node* node = new Add(...);
node->accept(visitor);
```

What happens:

1. Virtual dispatch calls `Add::accept()` because the actual object is an `Add`.
2. Inside `Add::accept()`, `this` has type `Add*`.
3. Therefore, `v.visit(this)` selects `visit(Add*)`, and virtual dispatch calls the concrete visitor's override.

So `accept()` is effectively a mechanism for **turning runtime type information into compile-time overload selection**.

### Could we eliminate `accept()` entirely?

Yes, using alternatives such as:

- `dynamic_cast` to identify the concrete node type.
- `std::variant` + `std::visit` if the node types form a closed set.
- A manually maintained node-kind enum and a `switch`.
- A dispatch table indexed by node type.

But the Visitor Pattern gives you that dispatch without manual type checks.

In Quadrants, [`BasicStmtVisitor::visit(Block*)`](https://github.com/Genesis-Embodied-AI/quadrants/blob/main/quadrants/ir/basic_stmt_visitor.cpp) visits statements through `Stmt*` pointers by calling `stmt->accept(this)`. That is a concrete example of why `accept()` is needed: it reaches the right typed `visit` overload even though the caller only has a base-class pointer.
