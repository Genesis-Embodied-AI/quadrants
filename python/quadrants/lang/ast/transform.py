# type: ignore

from quadrants.lang.ast.ast_transformer import ASTTransformer
from quadrants.lang.ast.ast_transformer_utils import ASTTransformerFuncContext


def transform_tree(tree, ctx: ASTTransformerFuncContext):
    ASTTransformer()(ctx, tree)
    return ctx.return_data
