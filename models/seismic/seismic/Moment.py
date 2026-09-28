# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# get the package
import altar
# and my base class
from altar.distributions.Uniform import Uniform as uniform


# the declaration
class Moment(uniform, family="altar.models.seismic.moment"):
    """
    The prior for slips (D) conforming to a given moment magnitude scale
        Mw = (log10 M0 - 9.1)/1.5 (Kanamori), M0 = sum_i Mu_i A_i D_i

    A uniform prior on {support}; initial samples are drawn from a gaussian Mw spread over the
    patches by a flat dirichlet, and {moment_constraint} optionally adds a gaussian penalty on
    the sample's Mw to the prior. Set {reparameterize} for gradient-based samplers.

    My numerics live in {altar.models.seismic.native.Moment} (cpu) or
    {altar.models.seismic.cuda.Moment}.
    """


    # user configurable state
    area = altar.properties.array(default=[1.0])
    area.doc = "the area of each patch in km^2; one value if the same for all patches"

    area_patch_file = altar.properties.path(default=None)
    area_patch_file.doc = "a text file with the area of each patch in km^2, overriding {area}"

    Mu = altar.properties.array(default=[32.0])
    Mu.doc = "the shear modulus of each patch in GPa; one value if the same for all patches"

    Mw_mean = altar.properties.float(default=1.0)
    Mw_mean.doc = "the mean moment magnitude"

    Mw_sigma = altar.properties.float(default=0.5)
    Mw_sigma.doc = "the standard deviation of the moment magnitude"

    slip_sign = altar.properties.str(default="positive")
    slip_sign.validators = altar.constraints.isMember("positive", "negative")
    slip_sign.doc = "the sign of the initial slips, all positive or all negative"

    moment_constraint = altar.properties.bool(default=False)
    moment_constraint.doc = "whether to add a gaussian penalty on the sample's Mw to the prior"

    moment_constraint_factor = altar.properties.float(default=1.0)
    moment_constraint_factor.doc = "a factor scaling the strength of the moment constraint"

    # my implementations live in this package
    impl_package = "altar.models.seismic"


    # implementation details
    def _makeImpl(self):
        """
        Hand my implementation my own extra state, beyond what {Uniform} already copies over
        """
        impl = super()._makeImpl()
        for trait in ("area", "area_patch_file", "Mu", "Mw_mean", "Mw_sigma", "slip_sign",
                      "moment_constraint", "moment_constraint_factor"):
            setattr(impl, trait, getattr(self, trait))
        return impl


# end of file
