#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved
#

"""
Backend selection helpers.

These helpers keep CUDA imports lazy so CPU-only builds don't import GPU modules.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

_active = "cpu"


def active() -> str:
    """
    Return the active backend name.
    """
    return _active


def cuda_available() -> bool:
    """
    Return True when altar.cuda is importable.
    """
    if find_spec("cuda") is None:
        return False
    if find_spec("altar.cuda.ext.cudaaltar") is None:
        return False
    return True


def activate_cpu() -> None:
    """
    Activate the CPU backend.
    """
    global _active
    _active = "cpu"


def activate_cuda() -> None:
    """
    Activate the CUDA backend by importing the CUDA overlay packages.
    """
    global _active
    # avoid work if already active
    if _active == "cuda":
        return
    # skip if not available
    if not cuda_available():
        return
    # import the core cuda package
    import_module("altar.cuda")
    # register CUDA implementations that override the cpu defaults
    import_module("altar.cuda.distributions")
    import_module("altar.cuda.norms")
    import_module("altar.cuda.data")
    import_module("altar.cuda.bayesian")
    import_module("altar.cuda.models")
    # mark active
    _active = "cuda"


def select_backend(*, application) -> None:
    """
    Select and activate the backend based on the application job settings.
    """
    if application.job.gpus > 0:
        activate_cuda()
    else:
        activate_cpu()
