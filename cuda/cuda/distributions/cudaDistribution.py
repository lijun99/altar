# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# externals
import math
# get the package
import altar
import altar.cuda

# get the base
from altar.distributions.Base import Base


# the declaration
class cudaDistribution(Base, family="altar.distributions.cudadistribution"):
    """
    The base class for probability distributions
    """

    # user configurable state from its cpu superclass
    parameters = altar.properties.int()
    parameters.doc = "the number of model parameters that belong to me"

    offset = altar.properties.int(default=0)
    offset.doc = "the starting point of my parameters in the overall model state"

    # configuration
    @altar.export
    def initialize(self, rng):
        """
        Initialize with the given random number generator
        """
        # will recommend a framework change to use application instead of rng
        # some distribution might need info from application
        # e.g, cascaded need worker id
        # so, use cuInitialize instead
        return self

    @altar.export
    def verify(self, theta, mask):
        # to satisfy component requirement
        # use cuVerify instead
        return self

    # cuda methods
    def cuInitialize(self, application):
        """
        cuda specific initialization
        """
        self.idx_range = (self.offset, self.offset + self.parameters)
        self.device = application.controller.worker.device
        self.precision = application.job.gpuprecision
        return self

    def cuInitSample(self, theta, batch):
        """
        cuda process to initialize random samples
        """
        return self

    def cuVerify(self, theta, mask, batch):
        """
        cuda process to verify the validity of samples
        """
        return mask

    def cuEvalPrior(self, theta, prior, batch):
        """
        cuda process to compute the prior
        """
        return prior

    def cuPriorGradient(self, theta, prior, batch, index=None):
        """
        Fill my portion of {prior} with the gradient of d\log P(\theta)/d\theta_{index}
        """
        # default, assume 0
        return self

    # private data
    device = None
    idx_range = None
    precision = None

# end of file
