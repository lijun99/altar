# -*- python -*-
# -*- coding: utf-8 -*-
#
#
# (c) 2013-2024 parasim inc
# (c) 2010-2024 california institute of technology
# all rights reserved
#

# get the package
import altar
# the package
import altar
# my protocol
from .LangevinScheduler import LangevinScheduler as scheduler
# externals
import math

# the scheduler protocol
class ExpDecay(altar.component, family="altar.langevin.schedulers.logdecay", implements=scheduler):
    """
    An exponential decay form of the Langevin factor epsilon_t(t) = a exp(-b t)
    """

    a = altar.properties.float(default=1)
    a.doc = "a in step size formula \epsilon_t = a exp(-b t)"

    b = altar.properties.float(default=1)
    b.doc = "b in step size formula \epsilon_t = a exp(-b t)"



    # required behavior
    @altar.provides
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        # nothing to do
        return self

    @altar.provides
    def epsilon_t(self, t):
        """
        Return the epsilon_t at {t}
        """

        return self.a*math.exp(-self.b*t)



# end of file
