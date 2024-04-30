# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# get the package
import altar

# the scheduler protocol
class LangevinScheduler(altar.protocol, family="altar.langevin.schedulers"):
    """
    The protocol that all AlTar schedulers must implement
    """

    # required behavior
    @altar.provides
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """

    @altar.provides
    def epsilon_t(self, t):
        """
        Return the epsilon_t at {t}
        """

    # framework hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Supply a default implementation
        """
        # by default, use COV
        from .PowerDecay import PowerDecay as default
        # and return it
        return default

# end of file
