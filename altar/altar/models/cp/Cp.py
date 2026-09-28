# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the package
import altar


# the protocol
class Cp(altar.protocol, family="altar.models.cp"):
    """
    The model uncertainty C_p, added to the data covariance C_d so that the data likelihood
    uses C_chi = C_d + C_p: none, a fixed C_p, or one the model re-estimates as sampling proceeds
    """


    @altar.provides
    def initialize(self, model, application):
        """
        Set up, once the model's data and parameter sets are initialized
        """


    @altar.provides
    def update(self, model, annealer, step):
        """
        At the start of a walk at a new beta, update the model's C_chi if needed; return True if
        it changed, so the densities of {step} get recomputed
        """


    @altar.provides
    def apply(self, model, theta):
        """
        Outside of sampling, e.g. for the forward check: set the model's C_chi as it would be
        for the mean model {theta}
        """


    # framework hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        No model uncertainty, by default
        """
        from .NoCp import NoCp
        return NoCp


# end of file
