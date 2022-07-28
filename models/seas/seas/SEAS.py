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
from seqeas.subduction import Simulation


class SEAS(BayesianL2, family="altar.models.seas"):
    """
    Wrapper around the subduction simulation class provided by
    ``seqeas.subduction.Simulation``.
    """
    # configurable properties
    config_file = altar.properties.str()

    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model
        """

        # call the super class initialization
        # super class method loads and initializes dataobs
        super().initialize(application=application)

        # initialize simulation object
        self.sim = Simulation(self.config_file)

        # done
        return self

    def forwardModel(self, theta, prediction):
        """
        Forward SEAS model
        """

        # set rheology viscosity
        log10_alpha_1 = theta[0]
        self.sim.fault.upper_rheo.alpha_1 = 10 ** log10_alpha_1

        # run simulation
        sol = self.sim.run(show_pbar=False)

        # get surface displacements
        surf_disps = self.sim.get_surface_displacements(sol)

        # fill the predictions array with the residuals
        prediction[:] = surf_disps[:, -self.sim.n_cycsamples:].ravel() - self.dataobs.dataobs

        # all done
        return self
