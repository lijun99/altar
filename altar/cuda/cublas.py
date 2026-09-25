# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
altar's own {cublas} namespace: everything {pyre.cuda.cublas} already has, plus {axpy}, the
one level-1 convenience wrapper the bayesian state layer's {compute_posterior} needs
(posterior = prior + beta*data). A separate module rather than monkey-patching
{pyre.cuda.cublas} itself, so pyre's own namespace stays exactly what pyre published.
"""

import numpy
import pyre.cuda


# the real thing, everything below is layered on top of
_cublas = pyre.cuda.cublas


def axpy(alpha, x, y, batch=None, handle=None):
    """
    y[:batch] = alpha * x[:batch] + y[:batch], dispatching on {x}'s cell type; {x}/{y} may be
    {altar.cuda.array.Array} instances or bare {pyre.grid} grids
    """
    # a lazy import: {altar.cuda} is still mid-initialization the first time this module
    # loads (it does `from . import cublas`), but by the time anyone actually calls {axpy}
    # it's long done
    import altar.cuda

    handle = handle if handle is not None else altar.cuda.cublas_handle()
    xg = x.grid if hasattr(x, "grid") else x
    yg = y.grid if hasattr(y, "grid") else y
    n = batch if batch is not None else xg.shape[0]

    cell = numpy.asarray(xg).dtype.name
    fn = _cublas.daxpy if cell == "float64" else _cublas.saxpy
    fn(handle, n, alpha, xg, 1, yg, 1)
    return y


def __getattr__(name):
    # forward everything else (dtrmm, cusolver-shared enums, ...) to pyre's own cublas
    return getattr(_cublas, name)


# end of file
