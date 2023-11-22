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


class SEAS3D(cudaBayesian, family="altar.models.seas.cuda.seas3d"):
    """
    Wrapper around the subduction simulation class provided by
    ``seqeas.subduction3d.SubductionSimulation3D``.
    """
    # configurable properties
    config_file = altar.properties.str()
    max_batch = altar.properties.int(default=None)
    systems_batch = altar.properties.int(default=None)

    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model
        """

        # call the super class initialization
        # super class method loads and initializes dataobs
        # ask dataobs to create duplicated data vectors
        self.dataobs.provide_batched_data = True
        # the model will take care of the cd_inv scaling instead
        self.dataobs.merge_cd_to_data = False
        super().initialize(application=application)

        self.gpuprec = application.job.gpuprecision
        channel = self.info
        if self.max_batch is None:
            self.max_batch = application.job.chains
        if self.systems_batch is None:
            self.systems_batch = application.job.chains

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
        self.t_obs_sec = altar.cuda.vector(
            source=(self.sim.t_obs * SEC_PER_YEAR).astype(self.gpuprec, order="C", copy=False))
        ievents = ([self.sim.ix_break_joint[-2]]
                   + [i for i in self.sim.ix_eq_joint if i > self.sim.ix_break_joint[-2]]
                   + [self.sim.ix_break_joint[-1]])
        tevents = self.sim.t_eval_joint[ievents] - self.sim.t_eval_joint[ievents[0]]
        self.t_events = altar.cuda.vector(
            source=(tevents * SEC_PER_YEAR).astype(self.gpuprec, order="C", copy=False))
        self.delta_tau_bounded_indices = \
            altar.cuda.vector(source=self.sim.delta_tau_bounded_indices.astype("int32"))
        self.ix_eq_joint = \
            altar.cuda.vector(source=self.sim.ix_eq_joint.astype("int32"))
        self.K_inner_inner_onfault = altar.cuda.vector(
            source=np.ascontiguousarray(self.fault.K_inner_inner[:, :2, :, :2],
                                        dtype=self.gpuprec))
        K_inner_asperities_v_plate = self.sim.K_inner_asperities_v_plate.T.ravel()
        self.K_inner_asperities_v_plate = \
            altar.cuda.vector(source=np.ascontiguousarray(K_inner_asperities_v_plate,
                                                          dtype=self.gpuprec))
        v_plate_ddcs_proj_eff_inner = \
            self.sim.v_plate_ddcs_proj_eff[self.fault.s_inner, :].T.ravel()
        self.v_plate_ddcs_proj_eff_inner = \
            altar.cuda.vector(source=np.ascontiguousarray(v_plate_ddcs_proj_eff_inner,
                                                          dtype=self.gpuprec))
        state_init_arr = np.concatenate([np.zeros(2 * self.fault.inner_num_patches),
                                         np.log(self.sim.v_init / self.rheo.v_0).T.ravel()])
        self.state_init = altar.cuda.vector(
            source=state_init_arr.astype(self.gpuprec, order="C", copy=False))
        self.sim_state = altar.cuda.vector(
            shape=application.job.chains * self.sim.t_obs.size * self.fault.inner_num_patches * 4,
            dtype=self.gpuprec)
        G_surf = self.sim.G_surf[:, :, self.sim.fault.s_inner, :] \
            .transpose(3, 2, 1, 0) \
            .reshape(2 * self.fault.inner_num_patches, 3 * self.sim.n_observers)
        self.G_surf = altar.cuda.matrix(source=np.ascontiguousarray(G_surf, dtype=self.gpuprec))
        self.obs_disp = altar.cuda.matrix(
            shape=(application.job.chains, self.sim.t_obs.size * 3 * self.sim.n_observers),
            dtype=self.gpuprec)
        self.alpha_h = \
            altar.cuda.vector(shape=application.job.chains * self.fault.inner_num_patches,
                              dtype=self.gpuprec)
        self.delta_tau_div_alpha_h = altar.cuda.vector(
            shape=application.job.chains * self.sim.n_eq * self.fault.inner_num_patches * 2,
            dtype=self.gpuprec)

        # create CUDA model for all samples (need to reduce to batch size)
        ticks.append(perf_counter())
        if self.gpuprec == "float64":
            self.cmodel = libcudaseas.ratedependent.model_double()
        elif self.gpuprec == "float32":
            self.cmodel = libcudaseas.ratedependent.model_float()
        else:
            raise NotImplementedError
        channel.log(f"Running in {self.gpuprec} precision")
        self.cmodel.initialize(
            self.max_batch,
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
            self.dataobs.dataobs[:] -= self.sim.zero_obs_at_eq(surf_disps_locked).T.ravel()
            self.precomputed_locked_disps = True
        surf_disps_outer = get_surface_displacements(
            self.sim.outer_creep_slip, self.sim.G_surf[:, :, self.fault.s_outer, :])
        surf_disps_lower = get_surface_displacements(
            self.sim.lower_creep_slip, self.sim.G_surf[:, :, self.fault.s_lower, :])
        self.dataobs.dataobs[:] -= self.sim.zero_obs_at_eq(surf_disps_outer + surf_disps_lower
                                                           ).T.ravel()
        # after any change of dataobs, update to cuda objects is needed
        # self.dataobs.updateCovariance()

        # print timings
        ticks.append(perf_counter())
        channel.log(f"Initialized SEAS3D in {ticks[-1] - ticks[0]}s")
        # channel.log(f"\n(Configuration = {ticks[1] - ticks[0]}s, "
        #             f"Python instances = {ticks[2] - ticks[1]}s, "
        #             f"GPU allocations = {ticks[3] - ticks[2]}s, "
        #             f"CUDA instance = {ticks[4] - ticks[3]}s, "
        #             f"Farfield effects = {ticks[5] - ticks[4]}s)")

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
            # check for 100km to 1m conversion
            if key in ["mid_transition", "deep_transition", "boundary_width"]:
                val = val * 1e5
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
        rheos = [self.rheo_from_theta(theta.get_row(i).copy_to_host(type="numpy"))
                 for i in range(batch)]

        # create new simulation instances, reusing G_surf
        ticks.append(perf_counter())
        sims = [SubductionSimulation3D(**self.sim_dict, rheo=rheos[i],
                                       fault=self.fault, G_surf=self.sim.G_surf)
                for i in range(batch)]

        # create stacked versions of alpha_h and delta_tau_div_alpha
        ticks.append(perf_counter())
        alpha_h_vec_stacked = np.stack([s.alpha_h_vec.squeeze() for s in sims])
        self.alpha_h.copy_from_host(
            source=np.ascontiguousarray(alpha_h_vec_stacked, dtype=self.gpuprec))
        delta_tau_bounded_compressed_stacked = \
            np.stack([s.delta_tau_bounded_compressed for s in sims])
        delta_tau_bound_comp_div_alpha = \
            delta_tau_bounded_compressed_stacked / alpha_h_vec_stacked[:, None, :, None]
        delta_tau_bound_comp_div_alpha = np.concatenate(
            [delta_tau_bound_comp_div_alpha[:, :, :, 0],
             delta_tau_bound_comp_div_alpha[:, :, :, 1]],
            axis=2)
        self.delta_tau_div_alpha_h.copy_from_host(
            source=np.ascontiguousarray(delta_tau_bound_comp_div_alpha, dtype=self.gpuprec))

        # call CUDA forward model
        ticks.append(perf_counter())
        self.cmodel.forward_model_batch(self.alpha_h.data,
                                        self.delta_tau_div_alpha_h.data,
                                        self.G_surf.data,
                                        prediction.data,
                                        batch)

        # reset observations to zero at beginning and at earthquakes
        # TODO: do this on GPU
        temp = prediction.copy_to_host(type="numpy").reshape(
            -1, self.sim.t_obs.size, 3, self.sim.n_observers)
        slips_obs = np.logical_and(self.sim.t_obs.min() <= self.sim.eq_df.index,
                                   self.sim.t_obs.max() > self.sim.eq_df.index)
        n_slips_obs = slips_obs.sum()
        temp -= temp[:, 0, :, :][:, None, :, :]
        if slips_obs.sum() > 0:
            i_slips_obs = [np.argmax(self.sim.t_obs >= t_eq) for t_eq
                           in self.sim.eq_df.index.values[slips_obs]]
            for i in range(n_slips_obs):
                temp[:, i_slips_obs[i]:, :, :] -= temp[:, i_slips_obs[i], :, :][:, None, :, :]
        prediction.copy_from_host(temp)

        # add locked slip if it was varied for samples
        if not self.precomputed_locked_disps:
            ticks.append(perf_counter())
            surf_disps_locked = np.stack(
                [self.sim.zero_obs_at_eq(get_surface_displacements(
                    sims[i].locked_slip, self.sim.G_surf[:, :, self.fault.s_asperities, :]
                 )).T.ravel() for i in range(batch)], axis=0, dtype=self.gpuprec)
            prediction += altar.cuda.matrix(source=surf_disps_locked)

        # TODO subsample the CUDA forward model for each station according to its availability

        # log timings
        ticks.append(perf_counter())
        infostr = (f"Ran forwardModelBatched for {batch} samples in {ticks[-1] - ticks[0]}s "
                   f"(CUDA forward model = {ticks[4] - ticks[3]}s)")
        # infostr += (f"\n(Rheology instances = {ticks[1] - ticks[0]}s, "
        #             f"Simulation instances = {ticks[2] - ticks[1]}s, "
        #             f"Stacked parameters = {ticks[3] - ticks[2]}s, "
        #             f"CUDA forward model = {ticks[4] - ticks[3]}s")
        # if self.precomputed_locked_disps:
        #     infostr += ")"
        # else:
        #     infostr += f", Farfield effects = {ticks[5] - ticks[4]}s)"
        channel = self.info
        channel.log(infostr)

        # all done
        return prediction

    def cuEvalLikelihood(self, theta, likelihood, batch):
        """
        Compute the likelihood from my forward problem
        :param: theta - sampled parameters, matrix of (samples, parameters)
        :param: likelihood - computed likelihood, vector of (samples)
        :param: batch - number of samples to be computed
        """

        # get the data storage for data prediction or residual
        residuals = self.obs_disp

        # solve forward modeling in batches
        # get the max batch size and allocate temporary input/out for a batch
        max_batch_size = self.systems_batch
        parameters = theta.shape[1]
        theta_batch = altar.cuda.matrix(shape=(max_batch_size, parameters), dtype=self.gpuprec)
        likelihood_batch = altar.cuda.vector(shape=max_batch_size, dtype=self.gpuprec)

        # iterate over batches
        for system_start in range(0, batch, max_batch_size):
            # get the actual batch size
            batch_size = min(max_batch_size, batch-system_start)
            # copy theta (a tile)
            theta_batch.copytile(src=theta, src_start=(system_start, 0), shape=(batch_size, parameters))
            # call forward model to calculate the data prediction or its difference between dataobs
            self.forwardModelBatched(theta=theta_batch, prediction=residuals, batch=batch_size)

            # call data method to calculate the l2 norm
            print(residuals.shape, likelihood_batch.shape, batch_size)
            self.dataobs.cuEvalLikelihood(prediction=residuals, likelihood=likelihood_batch,
                                          residual=True, batch=batch_size)
            # copy likelihood to global
            likelihood.copytile(likelihood_batch, start=system_start, size=batch_size)

        # consider cd_inv as a constant
        cd_inv = self.dataobs.gcd_inv
        print(type(cd_inv), cd_inv)
        if isinstance(cd_inv, float):
            likelihood *= cd_inv
        else:
            raise NotImplementedError

        # debug likelihood
        print("data likelihood")
        likelihood.print()

        # return the likelihood
        return likelihood
