# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# the package
import altar
import altar.cuda

# use the cpu protocol
from altar.distributions.Distribution import Distribution as distribution
# get default
from .cudaDistribution import cudaDistribution as cudaDistribution


@altar.foundry(implements=distribution, tip="the cuda Uniform probability distribution")
def uniform():
    # grab the factory
    from .cudaUniform import cudaUniform as uniform
    # attach its docstring
    __doc__ = uniform.__doc__
    # and return it
    return uniform

@altar.foundry(implements=distribution, tip="the cuda Uniform Logit probability distribution")
def uniformlogit():
    # grab the factory
    from .cudaUniformLogit import cudaUniformLogit as uniformlogit
    # attach its docstring
    __doc__ = uniformlogit.__doc__
    # and return it
    return uniformlogit



@altar.foundry(implements=distribution, tip="the cuda gaussian probability distribution")
def gaussian():
    # grab the factory
    from .cudaGaussian import cudaGaussian as gaussian
    # attach its docstring
    __doc__ = gaussian.__doc__
    # and return it
    return gaussian

@altar.foundry(implements=distribution, tip="the cuda truncated gaussian probability distribution")
def tgaussian():
    # grab the factory
    from .cudaTGaussian import cudaTGaussian as tgaussian
    # attach its docstring
    __doc__ = tgaussian.__doc__
    # and return it
    return tgaussian

@altar.foundry(implements=distribution, tip="the preset distribution")
def preset():
    # grab the factory
    from .cudaPreset import cudaPreset as preset
    # attach its docstring
    __doc__ = preset.__doc__
    # and return it
    return preset

# end of file
