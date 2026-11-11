# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# the package
import altar
# my base class
from altar.distributions.Base import Base as base


# declaration
class Pooled(base, family="altar.models.gnss.pooled"):
    """
    A sparsity prior on a group of basis functions of a trajectory case (e.g. the I-splines of
    one scale), pooled in space: for each column j, station s and pool of components (e.g. east
    and north), the energy of the column at the neighboring stations k,
        e = sum_k W_sk sum_{c in pool} theta_kcj^2,
    W a gaussian kernel of the distance ({length}, cut off at {radius}), and
        log P = - sum_{s, pool, j} phi(sqrt(e + epsilon^2)) + constant,
    with phi(r) = r / scale (shape "laplace": an overlapping group lasso, convex) or
    phi(r) = strength log(1 + r / scale) (shape "log": the penalty that reweighted L1, as in
    DISSTANS, minimizes). The columns of W are scaled so that, under the "laplace" shape, a
    coefficient active alone costs |theta| / scale, as under a plain Laplace prior (under "log",
    for small |theta|); active together at neighboring stations, coefficients cost less. The
    constant is unknown, so the evidence is off by it.

    My parameters are the group's, block by block; the blocks, their stations, components and
    coordinates are read from the case file {design_file}. Unbounded and smooth, for the
    gradient-based samplers; initial samples, if i am also the {prep}, are independent Laplace
    draws of {scale}.
    """


    # user configurable state
    design_file = altar.properties.path(default="trajectory.h5")
    design_file.doc = "the case file, for the stations, components and coordinates of my blocks"

    scale = altar.properties.float(default=1.0)
    scale.doc = "b, the scale of the penalty, in the units of the coefficients"

    shape = altar.properties.str(default="laplace")
    shape.doc = "phi(r): 'laplace', r / scale, or 'log', strength log(1 + r / scale)"
    shape.validators = altar.constraints.isMember("laplace", "log")

    strength = altar.properties.float(default=1.0)
    strength.doc = "the multiplier of the 'log' shape"

    epsilon = altar.properties.float(default=0.1)
    epsilon.doc = "the smoothing of the penalty at zero, in the units of the coefficients"

    length = altar.properties.float(default=30.0)
    length.doc = "the length (km) of the gaussian kernel of the neighborhood weights"

    radius = altar.properties.float(default=75.0)
    radius.doc = "the distance (km) beyond which stations aren't neighbors"

    pooling = altar.properties.str(default="horizontal")
    pooling.doc = "the components that share a sparsity pattern: 'components' (each alone), " \
                  "'horizontal' (east with north), 'all'"
    pooling.validators = altar.constraints.isMember("components", "horizontal", "all")


    # my implementations live in my own package
    impl_package = "altar.models.gnss"


    # implementation details
    def _makeImpl(self):
        """
        Hand my implementation my settings
        """
        impl = super()._makeImpl()
        for name in ("design_file", "scale", "shape", "strength", "epsilon", "length", "radius",
                     "pooling"):
            setattr(impl, name, getattr(self, name))
        return impl


# end of file
