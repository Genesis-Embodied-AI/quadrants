# type: ignore

from quadrants.lang.ast.ast_transformer import ASTTransformer
from quadrants.lang.ast.ast_transformer_utils import FuncCompilePassContext


def transform_tree(tree, ctx: FuncCompilePassContext):
    ASTTransformer()(ctx, tree)
    return ctx.return_data
