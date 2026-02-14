# Scalar tensors

To create a scalar tensor, use a shape of `()`. Note that this only works for fields, not ndarrays. (An NDArray needs to have at least one dimension).

You can access the value in a scalar tensor by indexing with `[None]`:

```
@ti.kernel
def f1(a: ti.Template) -> None:
    a[None] += 1
```
