# type: ignore

# The reason we import just the quadrants.core.util module, instead of the qd_python_core
# object within it, is that qd_python_core is stateful. While in practice qd_python_core is
# loaded during the import procedure, it's probably still good to delay the
# access to it.

from quadrants.lang import impl
from quadrants.lang.exception import QuadrantsRuntimeError


class SNodeTree:
    def __init__(self, ptr):
        self.prog = impl.get_runtime().prog
        self.ptr = ptr
        self.destroyed = False

    def destroy(self):
        if self.destroyed:
            raise QuadrantsRuntimeError("SNode tree has been destroyed")
        if self.prog != impl.get_runtime().prog:
            return
        self.ptr.destroy_snode_tree(impl.get_runtime().prog)

        # FieldExpression holds a SNode* to the place-SNode associated with a SNodeTree
        # Therefore, we have to recompile all the kernels after destroying a SNodeTree
        impl.get_runtime().clear_compiled_functions()
        self.destroyed = True

    @property
    def id(self):
        if self.destroyed:
            raise QuadrantsRuntimeError("SNode tree has been destroyed")
        return self.ptr.id()
