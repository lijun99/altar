# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# from pyre, imported here only, first, so that my modules reach them through me: the device
# manager, the cusolver bindings, {synchronize} to wait for the current device to finish its
# work, and {managed} to allocate a grid over a fresh block of managed memory
from pyre.cuda import manager, cusolver, synchronize, managed
# and the cublas/curand bindings, which my own {cublas}/{curand} extend
from pyre.cuda import cublas as pyrecublas, curand as pyrecurand

# my compiled extension
from . import ext

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
    Return the device this process runs on: the current cuda device, which each worker picks
    from {job.gpuids}, so that several ranks on one host each get their own
    """
    import cuda.core
    return manager.devices[cuda.core.Device().device_id]


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
