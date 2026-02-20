import time

import quadrants as qd

t = time.time()
qd.init(arch=qd.cuda, print_kernel_llvm_ir_optimized=True)


@qd.kernel
def p():
    print(42)


p()

print(f"{time.time() - t:.3f} s")
qd.core.print_profile_info()
