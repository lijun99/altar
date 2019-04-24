# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# the package
import altar


# the protocol
class DataObs(altar.protocol, family="altar.data"):
    """
    The protocol that all AlTar norms must satify
    """


    # interface
    @altar.provides
    def initialize(self, application):
        """
        initialize data
        """

    # framework hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Provide a default norm in case the user hasn't selected one
        """
        # the default is {L2}
        from .DataL2 import DataL2 as default
        # make it accessible
        return default


# end of file
