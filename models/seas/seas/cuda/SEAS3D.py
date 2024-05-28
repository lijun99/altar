# -*- python -*-
# -*- coding: utf-8 -*-
#
# author(s): Tobias Köhne

# general imports
import io
from time import perf_counter
from copy import deepcopy
from contextlib import redirect_stdout
import numpy as np

# import altar
import altar
from altar.cuda.models.cudaBayesian import cudaBayesian
from altar.models.seas.ext import cudaseas as libcudaseas

# import the earthquake cycle simulator
from seqeas.subduction3d import (RateStateSteadyLogarithmic2D, Fault3D, SubductionSimulation3D,
                                 get_surface_displacements)


class SEAS3D(cudaBayesian, family="altar.models.seas.cuda.seas3d"):
    """
    Wrapper around the subduction simulation class provided by
    ``seqeas.subduction3d.SubductionSimulation3D``.
    """
    # configurable properties
    config_file = altar.properties.str()
    cuda_batch_size = altar.properties.int(default=None)
    cuda_batch_size.doc = \
        "max system/sample size to be processed by cuda in a batch, as limited by gpu memory"
    cuda_threads = altar.properties.int(default=0)
    verbose = altar.properties.bool(default=False)
    ref_station_indices = altar.properties.list(default=None)
    velocity_reference_index = altar.properties.int(default=-1)
    estimate_row_indices = altar.properties.list(default=None, schema=altar.properties.int())
    estimate_column_indices = altar.properties.list(default=None, schema=altar.properties.int())

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
        # total number of systems to process per task
        self.num_systems = application.job.chains
        # if the cuda_batch_size is not given, use the total number of systems
        # alternatively, use a memory estimate to compute
        if self.cuda_batch_size is None:
            self.cuda_batch_size = self.num_systems

        # parse configuration
        ticks = []
        ticks.append(self.sync_and_time())
        self.rheo_dict, self.fault_dict, self.sim_dict = \
            SubductionSimulation3D.read_config_file(self.config_file)

        # create reference simulation
        ticks.append(self.sync_and_time())
        with redirect_stdout(io.StringIO()) as init_output:
            self.rheo = RateStateSteadyLogarithmic2D(**self.rheo_dict)
            self.fault = Fault3D(**self.fault_dict)
            self.sim = SubductionSimulation3D(**self.sim_dict, rheo=self.rheo, fault=self.fault)
        if self.verbose:
            channel.log(f"Device {self.device.id}: Simulation object initialization output"
                        f"\n{init_output.getvalue()}")

        # create intiialization dictionary for fast recreation
        self.sim_dict_fast = self.sim_dict.copy()
        self.sim_dict_fast.update({
            "fault": self.fault,
            "G_surf": self.sim.G_surf,
            "v_init": self.sim.v_init,
            "eq_df": self.sim.eq_df,
            "eq_slip": self.sim.eq_slip,
            "slip_taper_vec": self.sim.slip_taper_vec,
            "slip_taper_vec_nonuni": self.sim.slip_taper_vec_nonuni,
            "delta_tau_unbounded": self.sim.delta_tau_unbounded,
            "delta_tau_unbounded_nonuni": self.sim.delta_tau_unbounded_nonuni,
            "delta_tau_taper": self.sim.delta_tau_taper,
            "delta_tau_taper_nonuni": self.sim.delta_tau_taper_nonuni,
            "locked_slip": self.sim.locked_slip,
            "delta_tau_bounded_compressed": self.sim.delta_tau_bounded_compressed,
            "delta_tau_bounded_indices": self.sim.delta_tau_bounded_indices})

        # read number of rows/columns of alpha_h
        self.alpha_h_mat_rows = self.rheo.num_bases_depth
        self.alpha_h_mat_cols = self.rheo.num_bases_horiz

        # get index subset of values to estimate
        if self.estimate_row_indices is None:
            self.ix_estim_row = list(range(self.alpha_h_mat_rows))
        else:
            assert all([i < self.alpha_h_mat_rows for i in self.estimate_row_indices]), \
                "'estimate_row_indices' contains indices larger than the number of rows: " \
                f"{self.estimate_row_indices} >= {self.alpha_h_mat_rows}"
            self.ix_estim_row = self.estimate_row_indices
        if self.estimate_column_indices is None:
            self.ix_estim_col = list(range(self.alpha_h_mat_cols))
        else:
            assert all([i < self.alpha_h_mat_cols for i in self.estimate_column_indices]), \
                "'estimate_column_indices' contains indices larger than the number of columns: " \
                f"{self.estimate_column_indices} >= {self.alpha_h_mat_cols}"
            self.ix_estim_col = self.estimate_column_indices
        self.theta_subset_indices = \
            np.ix_(self.ix_estim_row, self.ix_estim_col)
        self.n_estim_row = len(self.ix_estim_row)
        self.n_estim_col = len(self.ix_estim_col)
        self.ordered_psets_list = [f"log10_alpha_h_{i}" for i in range(self.n_estim_row)]

        # get number of unique earthquakes
        num_orig_eq = self.sim.delta_tau_bounded_compressed.shape[0]
        self.num_eq = num_orig_eq
        if self.sim.delta_tau_bounded_nonuni is not None:
            num_final_eq = self.sim.delta_tau_unbounded_nonuni.shape[0]
            self.num_eq += num_final_eq

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

        state_init_single = altar.cuda.vector(
            source=state_init_arr.astype(self.gpuprec, order="C", copy=False))

        # change state_init to keep several copies of initial state
        self.state_init = altar.cuda.matrix(
            shape=(self.cuda_batch_size, state_init_single.size),
            dtype=self.gpuprec)
        self.state_init.duplicateVector(src=state_init_single)

        G_surf = self.sim.G_surf[:, :, self.sim.fault.s_inner, :] \
            .transpose(3, 2, 1, 0) \
            .reshape(2 * self.fault.inner_num_patches, 3 * self.sim.n_observers)
        self.G_surf = altar.cuda.matrix(source=np.ascontiguousarray(G_surf, dtype=self.gpuprec))

        # simulate state - yeval in ode
        self.sim_state = altar.cuda.vector(
            shape=self.cuda_batch_size * self.sim.t_obs.size * self.fault.inner_num_patches * 4,
            dtype=self.gpuprec)

        # copy the boolean observation mask onto the GPU
        if self.dataobs.mask is None:
            self.obs_mask = altar.cuda.vector(
                source=np.ones(self.sim.t_obs.size * 3 * self.sim.n_observers
                               ).astype(bool, order="C"))
            channel.log(f"Device {self.device.id}: Assuming no masked values")
        else:
            self.obs_mask = altar.cuda.vector(source=self.dataobs.mask)
            channel.log(f"Device {self.device.id}: Loaded mask with "
                        f"{(~self.dataobs.mask).sum()} masked values")

        # load reference stations, if present
        if self.ref_station_indices is not None:
            self.i_stat_ref = altar.cuda.vector(
                source=np.asarray(self.ref_station_indices, dtype="int32"))
            self.n_stat_ref = len(self.ref_station_indices)
            channel.log(f"Device {self.device.id}: Found {self.n_stat_ref} reference stations")
        else:
            self.i_stat_ref = altar.cuda.vector(
                source=np.asarray([]).astype(dtype="int32", order="C"))
            self.n_stat_ref = 0
            channel.log(f"Device {self.device.id}: No reference stations used")

        # predicted observations
        self.obs_disp = altar.cuda.matrix(
            shape=(self.cuda_batch_size, self.sim.t_obs.size * 3 * self.sim.n_observers),
            dtype=self.gpuprec)

        # create CUDA model for all samples (need to reduce to batch size)
        ticks.append(self.sync_and_time())
        if self.gpuprec == "float64":
            self.cmodel = libcudaseas.ratedependent.model_double()
        elif self.gpuprec == "float32":
            self.cmodel = libcudaseas.ratedependent.model_float()
        else:
            raise NotImplementedError
        channel.log(f"Device {self.device.id}: Running in {self.gpuprec} precision")
        self.cmodel.initialize(
            self.cuda_batch_size,
            self.sim.n_cycles_max,
            self.sim.t_obs.size,
            self.t_obs_sec.data,
            self.sim.n_slips,
            self.num_eq,
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
            self.sim.n_observers,
            self.obs_mask.data,
            self.i_stat_ref.data,
            self.n_stat_ref,
            self.velocity_reference_index)

        # remove precomputed farfield effects on observations
        ticks.append(self.sync_and_time())
        if np.any(self.sim.T_rec_logsigma) or (not self.sim.enforce_v_plate):
            raise NotImplementedError
        surf_disps_outer = get_surface_displacements(
            self.sim.outer_creep_slip, self.sim.G_surf[:, :, self.fault.s_outer, :])
        surf_disps_lower = get_surface_displacements(
            self.sim.lower_creep_slip, self.sim.G_surf[:, :, self.fault.s_lower, :])
        obs_farfield = (surf_disps_outer + surf_disps_lower).T  # to change into CUDA ordering
        self.obs_farfield = altar.cuda.vector(
            source=obs_farfield.ravel().astype(dtype=self.gpuprec, order="C"))

        # initialize forward model variables on GPU
        ticks.append(self.sync_and_time())
        self.delta_tau_bounded_indices = \
            altar.cuda.vector(source=self.sim.delta_tau_bounded_indices.astype("int32"))
        if self.sim.delta_tau_bounded_nonuni is None:
            self.delta_tau_bounded_indices_final = self.delta_tau_bounded_indices
        else:
            delta_tau_bounded_indices_final = self.sim.delta_tau_bounded_indices.copy()
            delta_tau_bounded_indices_final[-num_final_eq:] = np.arange(num_orig_eq, self.num_eq)
            self.delta_tau_bounded_indices_final = \
                altar.cuda.vector(source=delta_tau_bounded_indices_final.astype("int32"))

        # print timings
        ticks.append(self.sync_and_time())
        channel.log(f"Device {self.device.id}: Initialized SEAS3D in {ticks[-1] - ticks[0]:.1f}s")
        # channel.log(f"\n(Configuration = {ticks[1] - ticks[0]:.1f}s, "
        #             f"Python instances = {ticks[2] - ticks[1]:.1f}s, "
        #             f"GPU allocations = {ticks[3] - ticks[2]:.1f}s, "
        #             f"CUDA instance = {ticks[4] - ticks[3]:.1f}s, "
        #             f"Farfield effects = {ticks[5] - ticks[4]:.1f}s)")

        # keep a timer instance to check time between forwardModelBatched calls
        self.timer_fmb = self.sync_and_time()

        # done
        return self

    def rheo_from_theta(self, theta_arr):
        """
        Create a new Rheology object from a NumPy theta array.
        """
        # loop over theta entries
        rheo_kw_args = deepcopy(self.rheo_dict)
        if self.psets_list == ["log10_alpha_h_mat"]:  # single pset for entire matrix
            rheo_kw_args["alpha_h_mat"][self.theta_subset_indices] = \
                10**theta_arr.reshape(self.alpha_h_mat_rows, -1)
        else:  # parse individual rows of theta
            n_theta_rows = len(self.psets_list)
            n_theta_cols = list(set([self.psets[name].count for name in self.psets_list]))
            assert len(n_theta_cols) == 1, "Different lengths of psets."
            n_theta_cols = n_theta_cols[0]
            assert (n_theta_rows, n_theta_cols) == (self.n_estim_row, self.n_estim_col), \
                f"Expected theta of shape {(self.n_estim_row, self.n_estim_col)}, got " \
                f"shape {(n_theta_rows, n_theta_cols)}."
            theta_out = np.full((n_theta_rows, n_theta_cols), np.NaN)
            assert len(self.psets_list) == n_theta_rows
            for irow, name in enumerate(self.ordered_psets_list):
                itheta = self.psets_list.index(name)
                theta_out[irow, :] = theta_arr[itheta * n_theta_cols:(itheta + 1) * n_theta_cols]
            rheo_kw_args["alpha_h_mat"][self.theta_subset_indices] = 10**theta_out
        # return new object instance
        return RateStateSteadyLogarithmic2D(**rheo_kw_args)

    def forwardModelBatched(self, theta, prediction, batch_size_run):
        """
        Linear Viscous forward model in batch_size_run
        :param theta: matrix (batch_size_run, parameters), sampling parameters
        :param prediction: matrix (samples, observations), the predicted data or residual
                           between predicted and observed data
        :param batch_size_run: integer, the number of samples to be computed
                               batch_size_run<=samples
        :return: prediction as predicted data
        """

        # print info
        channel = self.info
        # channel.log(f"Device {self.device.id}: Time between forwardModelBatched calls: "
        #             f"{self.sync_and_time() - self.timer_fmb}s")

        # create new rheology instances
        ticks = []
        ticks.append(self.sync_and_time())
        rheos = [self.rheo_from_theta(theta.get_row(i).copy_to_host(type="numpy"))
                 for i in range(batch_size_run)]

        # create new simulation instances, reusing G_surf
        ticks.append(self.sync_and_time())
        sims = [SubductionSimulation3D(**self.sim_dict_fast, rheo=rheos[i])
                for i in range(batch_size_run)]

        # create stacked versions of alpha_h and delta_tau_div_alpha
        ticks.append(self.sync_and_time())
        alpha_h = altar.cuda.vector(
            shape=batch_size_run * self.fault.inner_num_patches, dtype=self.gpuprec)
        delta_tau_div_alpha_h = altar.cuda.vector(
            shape=(batch_size_run * self.num_eq * self.fault.inner_num_patches * 2),
            dtype=self.gpuprec)
        alpha_h_vec_stacked = np.stack([s.alpha_h_vec.squeeze() for s in sims])
        alpha_h.copy_from_host(
            source=np.ascontiguousarray(alpha_h_vec_stacked, dtype=self.gpuprec))
        if self.sim.delta_tau_bounded_nonuni is None:
            dtau_bound_comp_list = [s.delta_tau_bounded_compressed for s in sims]
        else:
            dtau_bound_comp_list = [np.concatenate([s.delta_tau_bounded_compressed,
                                                    s.delta_tau_bounded_nonuni], axis=0)
                                    for s in sims]
        delta_tau_bounded_compressed_stacked = np.stack(dtau_bound_comp_list)
        delta_tau_bound_comp_div_alpha = \
            delta_tau_bounded_compressed_stacked / alpha_h_vec_stacked[:, None, :, None]
        delta_tau_bound_comp_div_alpha = np.concatenate(
            [delta_tau_bound_comp_div_alpha[:, :, :, 0],
             delta_tau_bound_comp_div_alpha[:, :, :, 1]],
            axis=2)
        delta_tau_div_alpha_h.copy_from_host(
            source=np.ascontiguousarray(delta_tau_bound_comp_div_alpha,
                                        dtype=self.gpuprec))
        obs_ref = altar.cuda.matrix(shape=(batch_size_run, self.sim.t_obs.size * 3),
                                    dtype=self.gpuprec)

        # call CUDA forward model
        ticks.append(self.sync_and_time())
        self.cmodel.forward_model_batch(alpha_h.data,
                                        delta_tau_div_alpha_h.data,
                                        self.delta_tau_bounded_indices.data,
                                        self.delta_tau_bounded_indices_final.data,
                                        self.G_surf.data,
                                        prediction.data,
                                        obs_ref.data,
                                        self.obs_farfield.data,
                                        batch_size_run,
                                        self.cuda_threads,
                                        self.verbose)

        # log timings
        ticks.append(self.sync_and_time())
        infostr = (f"Device {self.device.id}: Ran forwardModelBatched for {batch_size_run} "
                   f"samples in {ticks[-1] - ticks[0]:.1f}s (")
        infostr += (f"Rheology instances = {ticks[1] - ticks[0]:.1f}s, "
                    f"Simulation instances = {ticks[2] - ticks[1]:.1f}s, "
                    f"Stacked parameters = {ticks[3] - ticks[2]:.1f}s, ")
        infostr += f"CUDA forward model = {ticks[4] - ticks[3]:.1f}s)"
        channel.log(infostr)

        # all done
        self.timer_fmb = self.sync_and_time()
        alpha_h.free()
        delta_tau_div_alpha_h.free()
        obs_ref.free()
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

        # # get logger
        # channel = self.info

        # solve forward modeling in batches
        # get the max batch size and allocate temporary input/out for a batch
        cuda_batch_size = self.cuda_batch_size
        parameters = theta.shape[1]
        # input
        theta_batch = altar.cuda.matrix(shape=(cuda_batch_size, parameters), dtype=self.gpuprec)
        # output
        likelihood_batch = altar.cuda.vector(shape=cuda_batch_size, dtype=self.gpuprec)

        # iterate over batches
        for system_start in range(0, batch, cuda_batch_size):
            # get the actual batch size
            batch_size_run = min(cuda_batch_size, batch - system_start)
            # channel.log(f"Device {self.device.id}: cuEvalLikelihood loop processing systems "
            #             f"{system_start} to {system_start + batch_size_run - 1}")
            # copy theta (a tile)
            theta_batch.copytile(src=theta,
                                 src_start=(system_start, 0),
                                 shape=(batch_size_run, parameters))
            # call forward model to calculate the data prediction
            self.forwardModelBatched(
                theta=theta_batch, prediction=predictions, batch_size_run=batch_size_run)
            # compute the residual
            data_obs = self.dataobs.gDataVec
            observations = data_obs.shape
            predictions.subtractVector(vector=data_obs, size=(batch_size_run, observations))
            # call data method to calculate the l2 norm
            self.dataobs.cuEvalLikelihood(prediction=predictions, likelihood=likelihood_batch,
                                          residual=True, batch=batch_size_run)
            # copy likelihood to global
            likelihood.copytile(likelihood_batch, start=system_start, size=batch_size_run)

        # consider cd_inv as a constant
        cd_inv = self.dataobs.gcd_inv
        if isinstance(cd_inv, float):
            likelihood *= cd_inv
        else:
            raise NotImplementedError

        theta_batch.free()
        likelihood_batch.free()

        # return the likelihood
        return likelihood
