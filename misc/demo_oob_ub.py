# https://forum.quadrants.graphics/t/quadrants/1003
import quadrants as qd

qd.init(arch=qd.cpu)

N = 3

x = qd.field(qd.i32, N)


@qd.kernel
def test():
    for i in x:
        x[i] = 1000 + i
    for i in qd.static(range(-N, 2 * N)):
        print(i, x[i])


test()
