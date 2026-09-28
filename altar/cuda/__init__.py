# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# export my parts
from . import (
    # norms
    norms,
    # probability distribution functions
    distributions,
    models,
    data,
    ext,
    )

# device management, and the thin cublas/cusolver/curand bindings, all from pyre now; the old
# top-level "cuda" package (a hand rolled device manager sharing its name with nvidia's own
# cuda-python) is gone -- see pyre's own "cuda: replace the hand-rolled device management
# extension with cuda-python" and the cublas/cusolver/curand work that followed it
import pyre.cuda

manager = pyre.cuda.manager
cusolver = pyre.cuda.cusolver

# altar's own {cublas}/{curand}: everything pyre's own has, plus a couple of convenience
# wrappers the bayesian sampler layer needs (see cublas.py/curand.py)
from . import cublas
from . import curand

# the pyre.grid-backed replacement for the old capsule-based cuda.Matrix/cuda.Vector (see
# array.py); the cuda extension's own kernels (norms, distributions, the samplers) take
# pyre.grid grids directly with no wrapper needed on that side -- {vector}/{matrix} exist for
# the bayesian state/sampler layer, which still calls them the way it always has
from .array import vector, matrix

# my extension modules
from .ext import cudaaltar as libcudaaltar


def get_current_device():
    """
    Return the device this process runs on; altar assumes one gpu per process (or per rank,
    under mpi), so this is always the first one pyre.cuda found
    """
    return manager.devices[0]


def curand_generator():
    """
    The curand generator cached on the current device, allocated once and reused; see
    {pyre.cuda.Device.curandGenerator}
    """
    return get_current_device().curandGenerator()


def cublas_handle():
    """
    The cublas handle cached on the current device, allocated once and reused; see
    {pyre.cuda.Device.cublasHandle}
    """
    return get_current_device().cublasHandle


def cusolver_handle():
    """
    The cusolver handle cached on the current device, allocated once and reused; see
    {pyre.cuda.Device.cusolverHandle}
    """
    return get_current_device().cusolverHandle


# administrative
def copyright():
    """
    Return the altar copyright note
    """
    return print(libcudaaltar.copyright())


def license():
    """
    Print the altar license
    """
    # print it
    return print(libcudaaltar.license())


def version():
    """
    Return the altar version
    """
    return libcudaaltar.version()

# end of file
