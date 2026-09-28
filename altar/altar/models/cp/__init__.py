# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import numpy
# the package
import altar
# the protocol
from .Cp import Cp as cp


@altar.foundry(implements=cp, tip="no model uncertainty")
def none():
    from .NoCp import NoCp
    __doc__ = NoCp.__doc__
    return NoCp


@altar.foundry(implements=cp, tip="a fixed, user-supplied model uncertainty")
def fixed():
    from .Fixed import Fixed
    __doc__ = Fixed.__doc__
    return Fixed


@altar.foundry(implements=cp, tip="a model uncertainty re-estimated from the posterior mean")
def adaptive():
    from .Adaptive import Adaptive
    __doc__ = Adaptive.__doc__
    return Adaptive


def sensitivity(model, cmu_file, kmu_file, predict):
    """
    C_p = K_p C_mu K_p^T for uncertain model inputs mu with covariance C_mu: column i of K_p is
    {predict}(K_i), the prediction with the green's functions replaced by the sensitivity kernel
    K_i = dG/dmu_i, read from the datasets of {kmu_file} in natural name order
    """
    import re
    import h5py
    cmu = numpy.atleast_2d(numpy.asarray(model.io.load(filename=cmu_file, dtype="float64")))
    order = lambda key: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", key)]
    with h5py.File(model.ifs[str(kmu_file)].uri.path, "r") as h5:
        keys = sorted(h5.keys(), key=order)
        if len(keys) != cmu.shape[0]:
            raise ValueError(f"'{kmu_file}' has {len(keys)} kernels, but C_mu is {cmu.shape}")
        Kp = numpy.column_stack([predict(numpy.asarray(h5[key], dtype="float64")) for key in keys])
    return Kp @ cmu @ Kp.T


# end of file
