# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# This is a copy of cudaLinear
# If used in cascaded model, make sure its parameters are the contiguous parameters in the beginning
# Otherwise, modifications of the following code are needed
# An easy way is to use self.restricted method to extract own parameters from theta

# the package
import altar
import altar.cuda
from altar.cuda import cublas
from altar.cuda import libcuda
from altar.cuda.models.cudaBayesianEnsemble import cudaBayesianEnsemble

# declaration
class cudaCascaded(cudaBayesianEnsemble, family="altar.models.seismic.cudacascaded"):
    """
    cascaded static/kinematic inversion
    """

    Npatches = altar.properties.int(default=1)
    Npatches.doc = "The total number of patches {Nas x Ndd}"
    

# end of file
