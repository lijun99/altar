# -*- python -*-
# -*- coding: utf-8 -*-
#
# proposal protocol for MCMC updates
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved
#

# get the package
import altar


# the proposal protocol
class Proposal(altar.protocol, family="altar.proposals"):
    """
    The protocol that all AlTar proposal mechanisms must implement
    """

    # required behavior
    @altar.provides
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """

    @altar.provides
    def propose(self, sampler, step, annealer=None):
        """
        Propose a new state by updating the samples in {step}
        """

    # framework hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Supply a default implementation
        """
        # by default, use a Gaussian proposal
        from .GaussianProposal import GaussianProposal as default
        # and return it
        return default


# end of file
