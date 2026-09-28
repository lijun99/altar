# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import numpy
# my base class
from .Base import Base as base
# the numerics shared with the cpu implementation
from ..native.Preset import load, rank, rows


# the declaration
class Preset(base):
    """
    The cuda implementation of preset samples; see {altar.distributions.Preset}
    """


    def initialize(self, rng, application=None):
        """
        The generic cuda-side setup, then find my file and read my samples
        """
        super().initialize(rng=rng, application=application)
        self.samples = load(distribution=self, application=application)
        self.rank = rank(application)
        return self


    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with my samples, this worker's share of them; a one-off host
        write into managed memory
        """
        θ = numpy.asarray(self._grid(theta))
        θ[:, self.idx_begin:self.idx_end] = rows(samples=self.samples, count=θ.shape[0], rank=self.rank)
        return self


    def verify(self, theta, mask, batch=None):
        """
        I am not a prior: nothing to check
        """
        return mask


    # private data, set by the shim before {initialize} runs
    input_file = None
    dataset = None
    # set by {initialize}
    samples = None
    rank = 0


# end of file
