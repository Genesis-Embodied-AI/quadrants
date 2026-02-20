# Sub-functions

A @qd.kernel can call other functions, as long as those functions have an appropriate taichi annotation.

## qd.func

@qd.func is the standard annotation for a function that can be called from a kernel. They can also be called from other @qd.func's. @qd.func is inlined into the calling kernel.

## qd.real_func

@qd.real_func is experimental. @qd.real_func is like @qd.func, but will not be inlined.
