# Unsupported python notations

## Argument unpacking

i.e. syntax like `some_function(*args)`

Argument unpacking is supported in a very narrow case, in order for test_utils in Genesis to work, https://github.com/Genesis-Embodied-AI/Genesis/blob/f711ef981a71558f8961f308066023d369405df4/tests/test_utils.py#L63:
- `*args` must be the last argument in the function call
- `*args` must not contain `dataclasses.dataclass` objects
