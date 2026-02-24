import quadrants as qd

qd.init(print_ir=True)

x = qd.field(qd.i32)
qd.root.dense(qd.i, 4).bitmasked(qd.i, 4).place(x)


@qd.kernel
def func():
    for i in x:
        print(i)


func()
