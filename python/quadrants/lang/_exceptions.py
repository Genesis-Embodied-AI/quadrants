from typing import Type


def raise_exception(ExceptionClass: Type[Exception], msg: str, err_code: str, orig: Exception | None = None):
    err = ExceptionClass(f"{msg} (error code: {err_code}).")
    if orig:
        raise err from orig
    raise err
