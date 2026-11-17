# type: ignore

from quadrants.lang.ast.ast_transformer import ASTTransformer
from quadrants.lang.ast.ast_transformer_utils import FuncCompileContext


def transform_tree(tree, ctx: FuncCompileContext):
    ASTTransformer()(ctx, tree)
    return ctx.return_data
