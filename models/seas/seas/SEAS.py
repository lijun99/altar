# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# author(s): Tobias Köhne

# import altar
import altar
from altar.models.BayesianL2 import BayesianL2

# import the earthquake cycle simulator
from seqeas.pyflat import SubductionSimulation


class SEAS(BayesianL2, family="altar.models.seas"):
    """
    Wrapper around the subduction simulation class provided by
    ``seqeas.pyflat.SubductionSimulation``.
    """
    # configurable properties
    config_file = altar.properties.str()
    slice_start = altar.properties.int()
    slice_end = altar.properties.int()
    slice_zero = altar.properties.int()

    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model
        """

        # call the super class initialization
        # super class method loads and initializes dataobs
        super().initialize(application=application)

        # initialize simulation object
        self.sim = SubductionSimulation(self.config_file)

        # done
        return self

    def forwardModel(self, theta, prediction):
        """
        Forward SEAS model
        """

        # set upper rheology
        alpha_eff = 10**theta[0]
        n = 10**theta[1]
        self.sim.set_upper_rheo_from_alpha_eff(alpha_eff=alpha_eff, n=n)

        # run simulation
        surf_disps = self.sim.zero_obs_at_eq(self.sim.run()[1])

        # fill the predictions array with the residuals
        prediction[:] = (surf_disps[:, self.slice_start:self.slice_end].ravel()
                         - self.dataobs.dataobs)

        # all done
        return self
