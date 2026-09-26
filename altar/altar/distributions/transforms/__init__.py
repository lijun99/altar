# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

import altar
from .Transform import Transform as transform

@altar.foundry(
    implements=transform,
    tip="a logit/expit transform between a bounded interval and an unconstrained real line")
def logittransform():
    from .Transform import LogitTransform
    __doc__ = LogitTransform.__doc__
    return LogitTransform

# end of file
