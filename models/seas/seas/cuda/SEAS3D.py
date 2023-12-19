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
    v_init_file = altar.properties.str(default=None)
    verbose = altar.properties.bool(default=False)

    # helper function to time
    def sync_and_time(self):
        self.device.synchronize()
        return perf_counter()

    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model
        """

        # call the super class initialization
        # super class method loads and initializes dataobs
        # ask dataobs not to create duplicated data vectors
        self.dataobs.provide_batched_data = False
        # the model will take care of the cd_inv scaling instead
        self.dataobs.merge_cd_to_data = False
        super().initialize(application=application)

        # additional preparations
        self.gpuprec = application.job.gpuprecision
        channel = self.info
        if self.max_batch is None:
            self.max_batch = application.job.chains
        if self.systems_batch is None:
            self.systems_batch = self.max_batch

        # parse configuration
        ticks = []
        ticks.append(self.sync_and_time())
        self.rheo_dict, self.fault_dict, self.sim_dict = \
            SubductionSimulation3D.read_config_file(self.config_file)

        # create reference simulation
        ticks.append(self.sync_and_time())
        self.rheo = RateStateSteadyLogarithmic(**self.rheo_dict)
        self.fault = Fault3D(**self.fault_dict)
        self.sim = SubductionSimulation3D(**self.sim_dict, rheo=self.rheo, fault=self.fault)

        # load initial velocity
        if self.v_init_file is not None:
            try:
                v_init_loaded = np.load(self.v_init_file)
                assert v_init_loaded.shape == \
                    self.sim.v_plate_ddcs_proj_eff[self.sim.fault.s_inner, :].shape
            except AssertionError:
                print("Couldn't load initial velocities due to shape mismatch.")
            except FileNotFoundError:
                print("Couldn't find initial velocities file.")
            else:
                self.sim.v_init = v_init_loaded
                print(f"Loaded initial velocities from '{self.v_init_file}'")

        # allocate memory
        ticks.append(self.sync_and_time())
        SEC_PER_YEAR = 86400 * 365.25
        self.t_obs_sec = altar.cuda.vector(
            source=(self.sim.t_obs * SEC_PER_YEAR).astype(self.gpuprec, order="C", copy=False))
        ievents = ([self.sim.ix_break_joint[-2]]
                   + [i for i in self.sim.ix_eq_joint if i > self.sim.ix_break_joint[-2]]
                   + [self.sim.ix_break_joint[-1]])
        tevents = self.sim.t_eval_joint[ievents] - self.sim.t_eval_joint[ievents[0]]
        self.t_events = altar.cuda.vector(
            source=(tevents * SEC_PER_YEAR).astype(self.gpuprec, order="C", copy=False))
        self.i_slips_obs = \
            altar.cuda.vector(source=np.array([0] + self.sim.i_slips_obs).astype("int32"))
        self.delta_tau_bounded_indices = \
            altar.cuda.vector(source=self.sim.delta_tau_bounded_indices.astype("int32"))
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

        G_surf = self.sim.G_surf[:, :, self.sim.fault.s_inner, :] \
            .transpose(3, 2, 1, 0) \
            .reshape(2 * self.fault.inner_num_patches, 3 * self.sim.n_observers)
        self.G_surf = altar.cuda.matrix(source=np.ascontiguousarray(G_surf, dtype=self.gpuprec))

        # simulate state - yeval in ode
        self.sim_state = altar.cuda.vector(
            shape=self.max_batch * self.sim.t_obs.size * self.fault.inner_num_patches * 4,
            dtype=self.gpuprec)

        # predicted observations
        self.obs_disp = altar.cuda.matrix(
            shape=(self.max_batch, self.sim.t_obs.size * 3 * self.sim.n_observers),
            dtype=self.gpuprec)

        # create CUDA model for all samples (need to reduce to batch size)
        ticks.append(self.sync_and_time())
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
            self.t_events.data,
            self.i_slips_obs.data,
            self.sim.n_slips_obs + 1,
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
        ticks.append(self.sync_and_time())
        if np.any(self.sim.T_rec_logsigma) or (not self.sim.enforce_v_plate):
            raise NotImplementedError
        surf_disps_outer = get_surface_displacements(
            self.sim.outer_creep_slip, self.sim.G_surf[:, :, self.fault.s_outer, :])
        surf_disps_lower = get_surface_displacements(
            self.sim.lower_creep_slip, self.sim.G_surf[:, :, self.fault.s_lower, :])
        self.dataobs.dataobs[:] -= self.sim.zero_obs_at_eq(surf_disps_outer + surf_disps_lower
                                                           ).T.ravel()
        # after any change of dataobs, update to cuda objects is needed
        self.dataobs.updateCovariance()

        # print timings
        ticks.append(self.sync_and_time())
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
        ticks.append(self.sync_and_time())
        rheos = [self.rheo_from_theta(theta.get_row(i).copy_to_host(type="numpy"))
                 for i in range(batch)]

        # create new simulation instances, reusing G_surf
        ticks.append(self.sync_and_time())
        sims = [SubductionSimulation3D(**self.sim_dict, rheo=rheos[i],
                                       fault=self.fault, G_surf=self.sim.G_surf)
                for i in range(batch)]

        # create stacked versions of alpha_h and delta_tau_div_alpha
        ticks.append(self.sync_and_time())
        alpha_h_vec_stacked = np.stack([s.alpha_h_vec.squeeze() for s in sims])
        alpha_h = altar.cuda.vector(shape=batch * self.fault.inner_num_patches, dtype=self.gpuprec)
        alpha_h.copy_from_host(
            source=np.ascontiguousarray(alpha_h_vec_stacked, dtype=self.gpuprec))
        delta_tau_bounded_compressed_stacked = \
            np.stack([s.delta_tau_bounded_compressed for s in sims])
        delta_tau_bound_comp_div_alpha = \
            delta_tau_bounded_compressed_stacked / alpha_h_vec_stacked[:, None, :, None]
        delta_tau_bound_comp_div_alpha = np.concatenate(
            [delta_tau_bound_comp_div_alpha[:, :, :, 0],
             delta_tau_bound_comp_div_alpha[:, :, :, 1]],
            axis=2)
        delta_tau_div_alpha_h = altar.cuda.vector(
            shape=batch * self.sim.n_eq * self.fault.inner_num_patches * 2,
            dtype=self.gpuprec)
        delta_tau_div_alpha_h.copy_from_host(
            source=np.ascontiguousarray(delta_tau_bound_comp_div_alpha,
                                        dtype=self.gpuprec))

        # call CUDA forward model
        ticks.append(self.sync_and_time())
        self.cmodel.forward_model_batch(alpha_h.data,
                                        delta_tau_div_alpha_h.data,
                                        self.G_surf.data,
                                        prediction.data,
                                        batch,
                                        self.verbose)

        # TODO subsample the CUDA forward model for each station according to its availability

        # log timings
        ticks.append(self.sync_and_time())
        infostr = (f"Ran forwardModelBatched for {batch} samples in {ticks[-1] - ticks[0]}s "
                   f"(CUDA forward model = {ticks[4] - ticks[3]}s)")
        # infostr += (f"\n(Rheology instances = {ticks[1] - ticks[0]}s, "
        #             f"Simulation instances = {ticks[2] - ticks[1]}s, "
        #             f"Stacked parameters = {ticks[3] - ticks[2]}s, "
        #             f"CUDA forward model = {ticks[4] - ticks[3]}s)")
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
        predictions = self.obs_disp

        # solve forward modeling in batches
        # get the max batch size and allocate temporary input/out for a batch
        max_batch_size = self.max_batch
        parameters = theta.shape[1]
        # input
        theta_batch = altar.cuda.matrix(shape=(max_batch_size, parameters), dtype=self.gpuprec)
        # output
        likelihood_batch = altar.cuda.vector(shape=max_batch_size, dtype=self.gpuprec)

        # iterate over batches
        for system_start in range(0, batch, max_batch_size):
            # get the actual batch size
            batch_size = min(max_batch_size, batch-system_start)
            print(f"Python Loop Processing systems {system_start} to {system_start+batch_size-1} ... ...")
            # copy theta (a tile)
            theta_batch.copytile(src=theta,
                                 src_start=(system_start, 0),
                                 shape=(batch_size, parameters))
            # call forward model to calculate the data prediction
            self.forwardModelBatched(theta=theta_batch, prediction=predictions, batch=batch_size)
            # compute the residual
            data_obs = self.dataobs.gDataVec
            observations = data_obs.shape

            # print("data_obs", data_obs.shape)
            # print("data_pre")
            # predictions.print()

            predictions.subtractVector(vector=data_obs, size=(batch_size, observations))
            # call data method to calculate the l2 norm
            self.dataobs.cuEvalLikelihood(prediction=predictions, likelihood=likelihood_batch,
                                          residual=True, batch=batch_size)
            # copy likelihood to global
            likelihood.copytile(likelihood_batch, start=system_start, size=batch_size)

        # consider cd_inv as a constant
        cd_inv = self.dataobs.gcd_inv
        if isinstance(cd_inv, float):
            likelihood *= cd_inv
        else:
            raise NotImplementedError

        # return the likelihood
        return likelihood
