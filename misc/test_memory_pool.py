# This file is not part of standard tests since it uses too much GPU memory

import quadrants as qd

qd.init(arch=qd.cuda, debug=True)

res = 512

mask = qd.field(qd.i32)
val = qd.field(qd.f32)

qd.root.dense(qd.ijk, 512).place(mask)
block = qd.root.pointer(qd.ijk, 128).dense(qd.ijk, 4)
block.dense(qd.l, 128).place(val)


@qd.kernel
def load_inputs():
    for i, j, k in mask:
        for l in range(128):
            val[i, j, k, l] = 1


load_inputs()
