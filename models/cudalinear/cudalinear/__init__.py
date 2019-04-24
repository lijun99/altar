# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# the package
import altar


# implementations
@altar.foundry(implements=altar.models.model, tip="a linear model")
def cudalinear():
    # grab the factory
    from .cudaLinear import cudaLinear as cudalinear
    # attach its docstring
    __doc__ = cudalinear.__doc__
    # and return it
    return cudalinear


# end of file
