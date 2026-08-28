# type: ignore
"""The ``qd.graph`` namespace: the canonical home for the graph-structuring kernel constructs.

``qd.graph.do_while`` / ``qd.graph.parallel_context`` / ``qd.graph.parallel`` are the preferred spellings. The flat
``qd.graph_do_while`` / ``qd.graph_parallel_context`` / ``qd.graph_parallel`` names remain valid (they are the same
objects re-exported here) but are deprecated: using a flat name inside a kernel emits a ``DeprecationWarning`` at
compile time. Both spellings are recognized by the AST compiler (see
``quadrants/lang/ast/ast_transformers/graph_api.py``).
"""

from quadrants.lang.graph_parallel import graph_parallel as parallel
from quadrants.lang.graph_parallel import graph_parallel_context as parallel_context
from quadrants.lang.misc import graph_do_while as do_while

__all__ = ["do_while", "parallel", "parallel_context"]
