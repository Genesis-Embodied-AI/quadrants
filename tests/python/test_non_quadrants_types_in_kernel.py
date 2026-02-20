import quadrants as qd

from tests import test_utils


@test_utils.test(exclude=qd.amdgpu)
def test_subscript_user_classes_in_kernel():
    class MyList:
        def __init__(self, elements):
            self.elements = elements

        def __getitem__(self, index):
            return self.elements[index]

    @qd.kernel
    def func():
        for i in qd.static(range(3)):
            print(a[i])

    a = MyList([1, 2, 3])
    func()
