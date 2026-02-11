import quadrants as ti


def test_binding():
    ti.init()
    quadrants_lang = ti._lib.core
    print(quadrants_lang.BinaryOpType.mul)
    one = quadrants_lang.make_const_expr_int(ti.i32, 1)
    two = quadrants_lang.make_const_expr_int(ti.i32, 2)
    expr = quadrants_lang.make_binary_op_expr(quadrants_lang.BinaryOpType.add, one, two)
    print(quadrants_lang.make_global_store_stmt(None, None))
