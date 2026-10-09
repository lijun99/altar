#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
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
    Return True when altar.cuda is importable, which needs a gpu the cuda driver can see
    """
    if find_spec("cuda") is None:
        return False
    # finding the extension imports altar.cuda, whose device discovery fails without a gpu
    try:
        return find_spec("altar.cuda.ext.cudaaltar") is not None
    except Exception:
        return False


def activate_cpu() -> None:
    """
    Activate the CPU backend.
    """
    global _active
    _active = "cpu"


def activate_cuda() -> None:
    """
    Activate the CUDA backend by importing the cuda package.
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
