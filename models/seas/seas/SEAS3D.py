# -*- python -*-
# -*- coding: utf-8 -*-
#
# author(s): Tobias Köhne

# general imports
import numpy as np
import pandas as pd

# import altar
import altar
from altar.models.BayesianL2 import BayesianL2

# import the earthquake cycle simulator
from seqeas.subduction3d import SubductionSimulation


class SEAS3D(BayesianL2, family="altar.models.seas"):
    """
    Wrapper around the subduction simulation class provided by
    ``seqeas.subduction3d.SubductionSimulation3D``.
    """
    # configurable properties
    config_file = altar.properties.str()
    obs_loc_file = altar.properties.str()
    t_obs_file = altar.properties.str()
    only_horizontals = altar.properties.bool(default=False)

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
        self.pts_surf = obs_loc.values.ravel()
        self.t_obs = np.load(self.t_obs_file)

        # read simulation configuration dictionary
        self.config_dict = SubductionSimulation.read_config_file(self.config_file)

        # done
        return self

    def update_cfg_from_theta(self, cfg, theta):
        """
        Convert the theta array into a SubductionSimulation-compatible dictionary.
        """
        i = 0
        theta_arr = theta.ndarray(copy=False)
        for name in self.psets_list:
            count = self.psets[name].count
            # get data
            val = theta_arr[i:i + count]
            # check whether we're setting a rheology
            if name.startswith("upper_"):
                key = name[6:]
                target = cfg["upper_rheo_kw_args"]
            elif name.startswith("lower_"):
                key = name[6:]
                target = cfg["lower_rheo_kw_args"]
            else:
                key = name
                target = cfg
            # check if we need to convert from log space
            if key.startswith("log10_"):
                val = 10**val
                key = key[6:]
            # check for km to m conversion
            if key in ["H", "mid_transition", "deep_transition", "deep_transition_width"]:
                val = val*1e3
            # apply update
            target[key] = val[0] if val.size == 1 else val.tolist()
            i += count
        # convert alpha_eff to alpha_n after we've set both alpha_eff and n
        for rheo in ["upper_rheo_kw_args", "lower_rheo_kw_args"]:
            try:
                alpha_eff = cfg[rheo].pop("alpha_eff")
            except (KeyError, AttributeError):
                pass
            else:
                cfg[rheo]["alpha_n"] = SubductionSimulation.get_alpha_n(
                    alpha_eff, cfg[rheo]["n"], cfg["v_plate"])
            try:
                alpha_eff_mid = cfg[rheo].pop("alpha_eff_mid")
            except (KeyError, AttributeError):
                pass
            else:
                n = cfg[rheo]["n_mid"] if "n_mid" in cfg[rheo] else cfg[rheo]["n"]
                cfg[rheo]["alpha_n_mid"] = SubductionSimulation.get_alpha_n(
                    alpha_eff_mid, n, cfg["v_plate"])
            try:
                alpha_eff_deep = cfg[rheo].pop("alpha_eff_deep")
            except (KeyError, AttributeError):
                pass
            else:
                if "n_deep" in cfg[rheo]:
                    n = cfg[rheo]["n_deep"]
                elif "n_mid" in cfg[rheo]:
                    n = cfg[rheo]["n_mid"]
                else:
                    n = cfg[rheo]["n"]
                cfg[rheo]["alpha_n_deep"] = SubductionSimulation.get_alpha_n(
                    alpha_eff_deep, n, cfg["v_plate"])
        return cfg

    def forwardModel(self, theta, prediction):
        """
        Forward SEAS model
        """
        # make a new configuration with updates from theta
        cfg = self.update_cfg_from_theta(self.config_dict.copy(), theta)

        # make and run simulation
        sim = SubductionSimulation.from_config_dict(cfg, self.t_obs, self.pts_surf)
        obs_zeroed = sim.zero_obs_at_eq(sim.run()[2])

        # restrict to horizontals if desired
        if self.only_horizontals:
            obs_zeroed = obs_zeroed[:obs_zeroed.shape[0]//2, :]

        # fill the predictions array with the residuals
        prediction[:] = obs_zeroed.ravel() - self.dataobs.dataobs

        # all done
        return self

    def forwardModelBatched(self, theta, prediction, batch):
        """
        Linear Viscous forward model in batch
        :param theta: matrix (samples, parameters), sampling parameters
        :param prediction: matrix (samples, observations), the predicted data or residual
                           between predicted and observed data
        :param batch: integer, the number of samples to be computed batch<=samples
        :return: prediction as predicted data
        """

        parameters = theta.shape[1]
        self.cmodel.forward_model(theta.data, prediction.data, parameters, batch)

        # all done
        return prediction
