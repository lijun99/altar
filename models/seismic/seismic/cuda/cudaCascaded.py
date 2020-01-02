# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# the package
import altar
import altar.cuda
from altar.cuda import cublas
from altar.cuda import libcuda
from altar.cuda.models.cudaBayesianEnsemble import cudaBayesianEnsemble

# declaration
class cudaCascaded(cudaBayesianEnsemble, family="altar.models.seismic.cuda.cascaded"):
    """
    Cascaded inversion of a model ensemble
    """


# end of file
