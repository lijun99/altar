# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# my base class
from .Base import Base as base


# declaration
class L2(base, family="altar.norms.l2"):
    """
    The L2 norm

    My actual numerics live in {altar.norms.native.L2.L2} (cpu) or
    {altar.norms.cuda.L2.L2}; see {Base} for how one gets picked.
    """


# end of file
