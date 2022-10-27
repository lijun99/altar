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

# general imports
import numpy as np
import pandas as pd

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
    obs_loc_file = altar.properties.str()
    t_obs_file = altar.properties.str()

    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model
        """

        # call the super class initialization
        # super class method loads and initializes dataobs
        super().initialize(application=application)

        # load ancillary data
        obs_loc = pd.read_csv(self.obs_loc_file, index_col=0)
        pts_surf = obs_loc.values.ravel()
        t_obs = pd.DatetimeIndex(np.load(self.t_obs_file))

        # initialize simulation object
        self.sim = SubductionSimulation(self.config_file, t_obs, pts_surf)

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
        obs_zeroed = self.sim.zero_obs_at_eq(self.sim.run()[2])

        # fill the predictions array with the residuals
        prediction[:] = (obs_zeroed[:self.sim.pts_surf.size, :].ravel() - self.dataobs.dataobs)

        # all done
        return self
