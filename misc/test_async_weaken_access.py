import quadrants as qd

qd.init()

x = qd.field(qd.i32)
y = qd.field(qd.i32)

qd.root.pointer(qd.ij, 4).dense(qd.ij, 8).place(x, y)


@qd.kernel
def copy():
    for i, j in y:
        x[i, j] = y[i, j]


copy()
