# -*- python -*-
# -*- coding: utf-8 -*-
#
# author(s): Tobias Köhne

# general imports
from time import perf_counter
from copy import deepcopy
import numpy as np

# import altar
import altar
from altar.cuda.models.cudaBayesian import cudaBayesian
from altar.models.seas.ext import cudaseas as libcudaseas

# import the earthquake cycle simulator
from seqeas.subduction3d import (RateStateSteadyLogarithmic, Fault3D, SubductionSimulation3D,
                                 get_surface_displacements)


class SEAS3D(cudaBayesian, family="altar.models.seas3d"):
    """
    Wrapper around the subduction simulation class provided by
    ``seqeas.subduction3d.SubductionSimulation3D``.
    """
    # configurable properties
    config_file = altar.properties.str()
    systems_batch = altar.properties.int()

    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model
        """

        # call the super class initialization
        # super class method loads and initializes dataobs
        super().initialize(application=application)

        # parse configuration
        ticks = []
        ticks.append(perf_counter())
        self.rheo_dict, self.fault_dict, self.sim_dict = \
            SubductionSimulation3D.read_config_file(self.config_file)

        # create reference simulation
        ticks.append(perf_counter())
        self.rheo = RateStateSteadyLogarithmic(**self.rheo_dict)
        self.fault = Fault3D(**self.fault_dict)
        self.sim = SubductionSimulation3D(**self.sim_dict, rheo=self.rheo, fault=self.fault)

        # allocate memory
        ticks.append(perf_counter())
        SEC_PER_YEAR = 86400 * 365.25
        self.t_obs_sec = altar.cuda.vector(source=self.sim.t_obs * SEC_PER_YEAR)
        ievents = ([self.sim.ix_break_joint[-2]]
                   + [i for i in self.sim.ix_eq_joint if i > self.sim.ix_break_joint[-2]]
                   + [self.sim.ix_break_joint[-1]])
        tevents = self.sim.t_eval_joint[ievents] - self.sim.t_eval_joint[ievents[0]]
        self.t_events = \
            altar.cuda.vector(source=tevents * SEC_PER_YEAR)
        self.delta_tau_bounded_indices = \
            altar.cuda.vector(source=self.sim.delta_tau_bounded_indices.astype("int32"))
        self.ix_eq_joint = \
            altar.cuda.vector(source=self.sim.ix_eq_joint.astype("int32"))
        self.K_inner_inner_onfault = \
            altar.cuda.vector(source=np.ascontiguousarray(self.fault.K_inner_inner[:, :2, :, :2]))
        K_inner_asperities_v_plate = self.sim.K_inner_asperities_v_plate.T.ravel()
        self.K_inner_asperities_v_plate = \
            altar.cuda.vector(source=np.ascontiguousarray(K_inner_asperities_v_plate))
        v_plate_ddcs_proj_eff_inner = \
            self.sim.v_plate_ddcs_proj_eff[self.fault.s_inner, :].T.ravel()
        self.v_plate_ddcs_proj_eff_inner = \
            altar.cuda.vector(source=np.ascontiguousarray(v_plate_ddcs_proj_eff_inner))
        state_init_arr = np.concatenate([np.zeros(2 * self.fault.inner_num_patches),
                                         np.log(self.sim.v_init / self.rheo.v_0).T.ravel()])
        self.state_init = altar.cuda.vector(source=state_init_arr)
        self.sim_state = altar.cuda.vector(
            shape=application.job.chains * self.sim.t_obs.size * self.fault.inner_num_patches * 4)
        G_surf = self.sim.G_surf[:, :, self.sim.fault.s_inner, :] \
            .transpose(3, 2, 1, 0) \
            .reshape(2 * self.fault.inner_num_patches, 3 * self.sim.n_observers)
        self.G_surf = altar.cuda.matrix(source=np.ascontiguousarray(G_surf))
        # self.obs_disp = altar.cuda.matrix(
        #     shape=(application.job.chains, self.sim.t_obs.size * 3 * self.sim.n_observers))
        # not necessary since prediction is deifned somewhere else?
        self.alpha_h = \
            altar.cuda.vector(shape=application.job.chains * self.fault.inner_num_patches)
        self.delta_tau_div_alpha_h = altar.cuda.vector(
            shape=application.job.chains * self.sim.n_eq * self.fault.inner_num_patches * 2)

        # create CUDA model for all samples (need to reduce to batch size)
        ticks.append(perf_counter())
        self.cmodel = libcudaseas.ratedependent.model_double()
        self.cmodel.initialize(
            application.job.chains,  # = max batch size
            self.systems_batch,
            self.sim.n_cycles_max,
            self.sim.t_obs.size,
            self.t_obs_sec.data,
            self.sim.n_slips,  # = tevents.size - 2
            self.sim.n_eq,
            self.delta_tau_bounded_indices.data,
            self.ix_eq_joint.data,
            self.t_events.data,
            self.rheo.v_0,
            self.fault.mu_over_2vs,
            self.fault.inner_num_patches,
            self.K_inner_inner_onfault.data,
            self.K_inner_asperities_v_plate.data,
            self.v_plate_ddcs_proj_eff_inner.data,
            self.state_init.data,
            self.sim_state.data,
            self.sim.atol,
            self.sim.rtol,
            self.sim.spinup_atol,
            self.sim.spinup_rtol,
            self.sim.n_observers)

        # remove precomputed farfield effects on observations
        ticks.append(perf_counter())
        if np.any(self.sim.T_rec_logsigma) or (not self.sim.enforce_v_plate):
            raise NotImplementedError
        if np.any(self.sim.D_0_logsigma):
            self.precomputed_locked_disps = False
        else:
            surf_disps_locked = get_surface_displacements(
                self.sim.locked_slip, self.sim.G_surf[:, :, self.fault.s_asperities, :])
            self.dataobs.dataobs[:] -= surf_disps_locked.T.ravel()
            self.precomputed_locked_disps = True
        surf_disps_outer = get_surface_displacements(
            self.sim.outer_creep_slip, self.sim.G_surf[:, :, self.fault.s_outer, :])
        surf_disps_lower = get_surface_displacements(
            self.sim.lower_creep_slip, self.sim.G_surf[:, :, self.fault.s_lower, :])
        self.dataobs.dataobs[:] -= (surf_disps_outer + surf_disps_lower).T.ravel()

        # print timings
        ticks.append(perf_counter())
        channel = self.info
        channel.log(f"Initialized SEAS3D in {ticks[-1] - ticks[0]}s\n"
                    f"(Configuration = {ticks[1] - ticks[0]}s, "
                    f"Python instances = {ticks[2] - ticks[1]}s, "
                    f"GPU allocations = {ticks[3] - ticks[2]}s, "
                    f"CUDA instance = {ticks[4] - ticks[3]}s, "
                    f"Farfield effects = {ticks[5] - ticks[4]}s)")

        # done
        return self

    def rheo_from_theta(self, theta_arr):
        """
        Create a new Rheology object from a NumPy theta array.
        """
        # loop over theta entries
        rheo_kw_args = deepcopy(self.rheo_dict)
        for i, name in enumerate(self.psets_list):
            # get value
            assert self.psets[name].count == 1
            key = name
            val = theta_arr[i]
            # check if we need to convert from log space
            if key.startswith("log10_"):
                val = 10**val
                key = key[6:]
            # check for km to m conversion
            if key in ["mid_transition", "deep_transition", "boundary_width"]:
                val = val * 1e3
            # apply update
            rheo_kw_args[key] = val
        # return new object instance
        return RateStateSteadyLogarithmic(**rheo_kw_args)

    def forwardModelBatched(self, theta, prediction, batch):
        """
        Linear Viscous forward model in batch
        :param theta: matrix (samples, parameters), sampling parameters
        :param prediction: matrix (samples, observations), the predicted data or residual
                           between predicted and observed data
        :param batch: integer, the number of samples to be computed batch<=samples
        :return: prediction as predicted data
        """

        # create new rheology instances
        ticks = []
        ticks.append(perf_counter())
        rheos = [self.rheo_from_theta(theta.getRow(i).ndarray(copy=False))
                 for i in range(batch)]

        # create new simulation instances, reusing G_surf
        ticks.append(perf_counter())
        sims = [SubductionSimulation3D(**self.sim_dict, rheo=rheos[i],
                                       fault=self.fault, G_surf=self.sim.G_surf)
                for i in range(batch)]

        # create stacked versions of alpha_h and delta_tau_div_alpha
        ticks.append(perf_counter())
        alpha_h_vec_stacked = np.stack([s.alpha_h_vec.squeeze() for s in sims])
        self.alpha_h.copy_from_host(source=np.ascontiguousarray(alpha_h_vec_stacked))
        delta_tau_bounded_compressed_stacked = \
            np.stack([s.delta_tau_bounded_compressed for s in sims])
        delta_tau_bound_comp_div_alpha = \
            delta_tau_bounded_compressed_stacked / alpha_h_vec_stacked[:, None, :, None]
        delta_tau_bound_comp_div_alpha = np.concatenate(
            [delta_tau_bound_comp_div_alpha[:, :, :, 0],
             delta_tau_bound_comp_div_alpha[:, :, :, 1]],
            axis=2)
        self.delta_tau_div_alpha_h.copy_from_host(
            source=np.ascontiguousarray(delta_tau_bound_comp_div_alpha))

        # call CUDA forward model
        ticks.append(perf_counter())
        self.cmodel.forward_model_batch(self.alpha_h.data,
                                        self.delta_tau_div_alpha_h.data,
                                        self.G_surf.data,
                                        prediction,
                                        batch)

        # add locked slip if it was varied for samples
        if not self.precomputed_locked_disps:
            ticks.append(perf_counter())
            surf_disps_locked = np.stack([get_surface_displacements(
                sims[i].locked_slip, self.sim.G_surf[:, :, self.fault.s_asperities, :]).T.ravel()
                for i in range(batch)], axis=0)
            prediction += altar.cuda.matrix(source=surf_disps_locked)

        # TODO subsample the CUDA forward model for each station according to its availability

        # log timings
        ticks.append(perf_counter())
        infostr = (f"Ran forwardModelBatched in {ticks[-1] - ticks[0]}s\n"
                   f"(Rheology instances = {ticks[1] - ticks[0]}s, "
                   f"Simulation instances = {ticks[2] - ticks[1]}s, "
                   f"Stacked parameters = {ticks[3] - ticks[2]}s, "
                   f"CUDA forward model = {ticks[4] - ticks[3]}s")
        if self.precomputed_locked_disps:
            infostr += ")"
        else:
            infostr += f", Farfield effects = {ticks[5] - ticks[4]}s)"
        channel = self.info
        channel.log(infostr)

        # all done
        return prediction
