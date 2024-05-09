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

# the scheduler protocol
class PowerDecay(altar.component, family="altar.langevin.schedulers.powerdecay", implements=scheduler):
    """
    A power-law decay form of the Langevin factor epsilon_t(t)
    """

    a = altar.properties.float(default=1)
    a.doc = "a in step size formula \epsilon_t = a/(b+t)^\gamma"

    b = altar.properties.float(default=1)
    b.doc = "b in step size formula \epsilon_t = a/(b+t)^\gamma"

    gamma = altar.properties.float(default=0.9)
    gamma.doc = "gamma in step size formula \epsilon_t = a/(b+t)^\gamma, \gamma \in (0.5, 1]"

    estimate_a = altar.properties.bool(default=False)

    estimate_scale = altar.properties.float(default=1)


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
        return self.a*pow(self.b+t, -self.gamma)

    def start(self, controller):
        """
        Processes to run before sampling
        """
        if self.estimate_a:
            # call worker method
            rate = controller.worker.estimate_rate(controller=controller, scale=self.estimate_scale)
            #
            self.a = rate

        channel = controller.info
        channel.log(f"starting sampling rate {self.a}")
        # all done
        return self

# end of file
