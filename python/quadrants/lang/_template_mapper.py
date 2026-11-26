"""Preserve the deprecated TemplateMapper import for existing callers."""

import warnings

from quadrants.lang._kernel_specialization_mapper import (
    KernelSpecializationMapper as TemplateMapper,
)

warnings.warn(
    "quadrants.lang._template_mapper.TemplateMapper is deprecated. "
    "Use quadrants.lang._kernel_specialization_mapper.KernelSpecializationMapper instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TemplateMapper"]
