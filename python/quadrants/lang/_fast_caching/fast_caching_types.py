from pydantic import BaseModel

from .._wrap_inspect import FunctionSourceInfo


class HashedFunctionSourceInfo(BaseModel):
    """
    Wraps a function source info, and the hash string of that function.

    By not adding the hash directly into function source info, we avoid
    having to make hash an optional type, and checking if it's empty or not.

    If you have a HashedFunctionSourceInfo object, then you are guaranteed
    to have the hash string.

    If you only have the FunctionSourceInfo object, you are guaranteed that it
    does not have a hash string.
    """

    function_source_info: FunctionSourceInfo
    hash: str
