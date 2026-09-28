# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# support
import altar

# the plexus
from .AlTar import AlTar


class cudaAlTar(AlTar, family="altar.shells.cudaaltar", namespace="altar"):
    """
    Backward-compatible alias for the unified AlTar shell.
    """


# end of file
