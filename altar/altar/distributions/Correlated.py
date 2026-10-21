# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import math
# get the package
import altar

# get the protocol
from . import distribution
# and my base class
from .Base import Base as base


# the declaration
class Correlated(base, family="altar.distributions.correlated"):
    """
    A gaussian distribution whose parameters are correlated in space, e.g. the slips of the
    patches of a fault: N(mean, C), with C_ij = sigma^2 k(|x_i - x_j| / length) for the
    locations x_i of the parameters, read from {coordinates}, and a correlation kernel k

    My actual numerics live in {altar.distributions.native.Correlated.Correlated} (cpu) or
    {altar.distributions.cuda.Correlated.Correlated}; see {Base} for how one gets picked.
    """


    # user configurable state
    mean = altar.properties.float(default=0)
    mean.doc = "the mean of each parameter"

    sigma = altar.properties.float(default=1)
    sigma.doc = "the standard deviation of each parameter"

    kernel = altar.properties.str(default="exponential")
    kernel.doc = "the correlation with distance r: exponential, exp(-r); matern32, " \
                 "(1 + √3 r) exp(-√3 r); or gaussian, exp(-r^2/2), r in units of {length}"

    length = altar.properties.array(default=(1.0,))
    length.doc = "the correlation length, in the units of {coordinates}: one value, or one " \
                 "per coordinate, e.g. (40, 20) along strike and down dip"

    coordinates = altar.properties.path(default=None)
    coordinates.doc = "the locations of my parameters, (parameters x dimensions), e.g. the " \
                      "centroids of the patches: an .h5 or a text file, relative to the " \
                      "current directory or to the model's input directory"

    dataset = altar.properties.str(default=None)
    dataset.doc = "the dataset of {coordinates} in an .h5 file; default the first one"

    support = altar.properties.array(default=(-math.inf, math.inf))
    support.doc = "the support of the distribution, e.g. (0, 20) to keep the slips positive; " \
                  "unbounded by default"

    reparameterize = altar.properties.bool(default=False)
    reparameterize.doc = \
        "whether to reparameterize a finite {support} to an unconstrained sampling space via " \
        "{transform}, for use by gradient-based samplers"

    transform = altar.distributions.transform()
    transform.doc = "the transform used to map between physical and sampling space, when " \
                    "{reparameterize} is set"

    jitter = altar.properties.float(default=1e-6)
    jitter.doc = "added to the diagonal of the correlation matrix, to keep it positive definite"

    sweeps = altar.properties.int(default=100)
    sweeps.doc = "the sweeps of elliptical slice sampling for the initial samples within a " \
                 "bounded {support}"


    # configuration
    @altar.export
    def initialize(self, rng, application=None):
        """
        Initialize, then mirror whether my support turned out bounded
        """
        super().initialize(rng=rng, application=application)
        self.bounded = self._impl.bounded
        return self


    # implementation details
    def _makeImpl(self):
        """
        Hand my implementation my own extra state, beyond what {Base} already copies over
        """
        impl = super()._makeImpl()
        for name in ["mean", "sigma", "kernel", "length", "coordinates", "dataset", "support",
                     "reparameterize", "transform", "jitter", "sweeps"]:
            setattr(impl, name, getattr(self, name))
        return impl


# end of file
