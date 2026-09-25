# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

import altar
from .Proposal import Proposal as proposal

@altar.foundry(
    implements=proposal,
    tip="a Gaussian proposal mechanism for sampling updates")
def gaussianproposal():
    from .GaussianProposal import GaussianProposal
    __doc__ = GaussianProposal.__doc__
    return GaussianProposal

# end of file
