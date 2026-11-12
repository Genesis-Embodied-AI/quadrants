import quadrants as qd


def qd_init_same_arch(**options) -> None:
    """
    Used in tests to call qd.init, passing in the same arch as currently
    configured. Since it's fairly fiddly to do that, extracting this out
    to this helper function.
    """
    assert qd.cfg is not None
    options = dict(options)
    options["arch"] = getattr(qd, qd.cfg.arch.name)
    qd.init(**options)
