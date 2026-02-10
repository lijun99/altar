# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#


# superclass
from .LangevinMethod import LangevinMethod


# declaration
class SequentialLangevin(LangevinMethod):
    """
    Implementation that assumes its state is the global state of the solver, and therefore it
    is able to compute the statistical properties of the sample distribution
    """


    # public data
    wid = 0     # my worker id
    workers = 1 # i don't manage anybody else


    # interface
    def start(self, controller):
        """
        Start the langevin process
        """
        # chain up
        super().start(controller=controller)
        # build a cooling step to hold the state of the problem
        self.step = self.CoolingStep.start(controller=controller)
        # all done
        return self


# end of file
