import quadrants as qd


def test_binding():
    qd.init()
    quadrants_lang = qd._lib.core
    print(quadrants_lang.BinaryOpType.mul)
    one = quadrants_lang.make_const_expr_int(qd.i32, 1)
    two = quadrants_lang.make_const_expr_int(qd.i32, 2)
    expr = quadrants_lang.make_binary_op_expr(quadrants_lang.BinaryOpType.add, one, two)
    print(quadrants_lang.make_global_store_stmt(None, None))
