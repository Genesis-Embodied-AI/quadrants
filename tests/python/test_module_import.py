import quadrants as myownquadrants

from tests import test_utils


@test_utils.test()
def test_module_import():
    @myownquadrants.kernel
    def func():
        for _ in myownquadrants.static(range(8)):
            pass

    func()
