# Sub-functions

A @ti.kernel can call other functions, as long as those functions have an appropriate taichi annotation.

## ti.func

@ti.func is the standard annotation for a function that can be called from a kernel. They can also be called from other @ti.func's. @ti.func is inlined into the calling kernel.

## ti.real_func

@ti.real_func is experimental. @ti.real_func is like @ti.func, but will not be inlined.
