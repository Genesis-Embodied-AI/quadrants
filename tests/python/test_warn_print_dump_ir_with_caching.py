import os
import warnings

import pytest

import quadrants as qd

WARNING_CODE = r"\[warning_code=DUMP_IR_CACHE_MISMATCH\]"


def test_warn_caching_with_print_ir():
    with pytest.warns(
        UserWarning,
        match=WARNING_CODE,
    ):
        qd.init(print_ir=True, log_level="warn", offline_cache=True)


def test_warn_caching_with_qd_dump():
    DUMP_IR_ENV_VAR = "QD_DUMP_IR"
    os.environ[DUMP_IR_ENV_VAR] = "1"

    with pytest.warns(
        UserWarning,
        match=WARNING_CODE,
    ):
        qd.init(log_level="warn", offline_cache=True)

    del os.environ[DUMP_IR_ENV_VAR]


def test_no_warn_caching_ir():
    warnings.filterwarnings("error")
    try:
        qd.init(log_level="warn", offline_cache=True)
    except UserWarning as user_warning:
        assert WARNING_CODE not in user_warning.args[0]

    warnings.resetwarnings()
