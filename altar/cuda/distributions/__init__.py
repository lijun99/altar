# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# the package
import altar
import altar.cuda

# use the cpu protocol 
from altar.distributions.Distribution import Distribution as distribution
# get default
from .cudaDistribution import cudaDistribution as cudaDistribution


@altar.foundry(implements=distribution, tip="the cuda cudaUniform probability distribution")
def uniform():
    # unlike gaussian/tgaussian below, this one is NOT redirected to the unified
    # altar.distributions.Uniform: models/seismic's cudaMoment still subclasses cudaUniform
    # directly, so the class has to keep existing until that model is migrated too
    # grab the factory
    from .cudaUniform import cudaUniform as uniform
    # attach its docstring
    __doc__ = uniform.__doc__
    # and return it
    return uniform

# {Gaussian}/{TGaussian} are unified components now: each picks its cpu/cuda implementation
# internally, at {initialize} time, so these two foundries exist only for {.pfg} files that
# still spell out the explicit "altar.cuda.distributions.gaussian"/"...tgaussian" path; they
# resolve to the very same classes {altar.distributions.gaussian}/{.tgaussian} do
@altar.foundry(implements=distribution, tip="the gaussian probability distribution")
def gaussian():
    # grab the factory
    from altar.distributions.Gaussian import Gaussian as gaussian
    # attach its docstring
    __doc__ = gaussian.__doc__
    # and return it
    return gaussian

@altar.foundry(implements=distribution, tip="the gaussian probability distribution, truncated to a finite support")
def tgaussian():
    # grab the factory
    from altar.distributions.TGaussian import TGaussian as tgaussian
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
