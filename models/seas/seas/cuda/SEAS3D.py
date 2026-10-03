# -*- python -*-
# -*- coding: utf-8 -*-
#
# author(s): Tobias Köhne

# general imports
import io
from time import perf_counter
from copy import copy, deepcopy
from contextlib import redirect_stdout
import numpy as np
import pyre.cuda

# import altar
import altar
import altar.cuda
from altar.models.BayesianL2 import BayesianL2
from altar.models.seas.ext import cudaseas as libcudaseas

# import the earthquake cycle simulator
from seqeas.subduction3d import (
    RateStateSteadyLogarithmic2D,
    Fault3D,
    SubductionSimulation3D,
    get_surface_displacements,
)


class SEAS3D(BayesianL2, family="altar.models.seas.seas3d"):
    """
    Wrapper around the subduction simulation class provided by
    ``seqeas.subduction3d.SubductionSimulation3D``; runs on the gpu only
    """

    # configurable properties
    config_file = altar.properties.str()
    cuda_batch_size = altar.properties.int(default=None)
    cuda_batch_size.doc = (
        "max system/sample size to be processed by "
        "cuda in a batch, as limited by gpu memory"
    )
    cuda_threads = altar.properties.int(default=0)
    cuda_threads.doc = (
        "threads per block for the integrator; 0 picks them from the number of "
        "patches; either way, capped by what the integrator's registers allow"
    )
    integrator = altar.properties.str(default="dopri5")
    integrator.validators = altar.constraints.isMember("dopri5", "radau5")
    integrator.doc = (
        "the ode integrator: dopri5 (explicit Runge-Kutta 5(4)), or radau5 "
        "(implicit Radau IIA, for stiff systems)"
    )
    gradient_step = altar.properties.float(default=1e-4)
    gradient_step.doc = (
        "the step of the central differences of the data likelihood gradient, "
        "in the units of the parameters"
    )
    verbose = altar.properties.bool(default=False)
    forward_ode = altar.properties.str(default="ratedependent")
    forward_ode.validators = altar.constraints.isMember(
        "ratedependent", "tractiondependent"
    )
    forward_ode.doc = (
        "which forward ODE to use, the simpler 'ratedependent' tracking logarithmic "
        "velocity or the 'tractiondependent' one which tracks the elastic traction"
    )
    ref_station_indices = altar.properties.list(default=None)
    velocity_reference_index = altar.properties.int(default=-1)
    estimate_row_indices = altar.properties.list(
        default=None, schema=altar.properties.int()
    )
    estimate_column_indices = altar.properties.list(
        default=None, schema=altar.properties.int()
    )

    # helper function to time
    def sync_and_time(self):
        pyre.cuda.synchronize()
        return perf_counter()

    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model
        """

        # the integrators are cuda only
        if altar.backends.active() != "cuda":
            application.error.log("seas3d runs on the gpu only; set job.gpus = 1")
            raise SystemExit(1)

        # call the super class initialization
        # super class method loads and initializes dataobs and the parameter sets
        super().initialize(application=application)

        # additional preparations
        self.gpuprec = self.precision
        self.device = altar.cuda.get_current_device()
        channel = self.info
        # if the cuda_batch_size is not given, use the total number of systems
        # alternatively, use a memory estimate to compute
        if self.cuda_batch_size is None:
            self.cuda_batch_size = self.samples

        # parse configuration
        ticks = []
        ticks.append(self.sync_and_time())
        self.rheo_dict, self.fault_dict, self.sim_dict = (
            SubductionSimulation3D.read_config_file(self.config_file)
        )

        # create reference simulation
        ticks.append(self.sync_and_time())
        with redirect_stdout(io.StringIO()) as init_output:
            self.rheo = RateStateSteadyLogarithmic2D(**self.rheo_dict)
            self.fault = Fault3D(**self.fault_dict)
            self.sim = SubductionSimulation3D(
                **self.sim_dict,
                rheo=self.rheo,
                fault=self.fault,
                calculate_tapered_slip=False,
            )
        if self.verbose:
            channel.log(
                f"Device {self.device.id}: Simulation object initialization output"
                f"\n{init_output.getvalue()}"
            )

        # create deep copies of sim that can later be easily modified for the forward runs
        self.sims_storage = [copy(self.sim) for _ in range(self.cuda_batch_size)]

        # read number of rows/columns of alpha_h
        self.alpha_h_mat_rows = self.rheo.num_bases_depth
        self.alpha_h_mat_cols = self.rheo.num_bases_horiz

        # get maximum integration velocity
        self.v_ratio_max = (
            0 if self.sim.v_max is None else self.sim.v_max / self.rheo.v_0
        )
        # the traction integrator needs rho
        if (self.forward_ode == "tractiondependent") and (self.sim.rho is None):
            self.error.log(
                "the traction-dependent integrator needs rho; "
                "set it in the simulation configuration"
            )
            raise SystemExit(1)

        # get index subset of values to estimate
        if self.estimate_row_indices is None:
            self.ix_estim_row = list(range(self.alpha_h_mat_rows))
        else:
            assert all(
                [i < self.alpha_h_mat_rows for i in self.estimate_row_indices]
            ), (
                "'estimate_row_indices' contains indices larger than the number of "
                f"rows: {self.estimate_row_indices} >= {self.alpha_h_mat_rows}"
            )
            self.ix_estim_row = self.estimate_row_indices
        if self.estimate_column_indices is None:
            self.ix_estim_col = list(range(self.alpha_h_mat_cols))
        else:
            assert all(
                [i < self.alpha_h_mat_cols for i in self.estimate_column_indices]
            ), (
                "'estimate_column_indices' contains indices larger than the number of "
                f"columns: {self.estimate_column_indices} >= {self.alpha_h_mat_cols}"
            )
            self.ix_estim_col = self.estimate_column_indices
        self.theta_subset_indices = np.ix_(self.ix_estim_row, self.ix_estim_col)
        self.n_estim_row = len(self.ix_estim_row)
        self.n_estim_col = len(self.ix_estim_col)
        self.ordered_psets_list = [
            f"log10_alpha_h_{i}" for i in range(self.n_estim_row)
        ]

        # get number of unique earthquakes
        num_orig_eq = self.sim.delta_tau_bounded_compressed.shape[0]
        self.num_eq = num_orig_eq
        if self.sim.delta_tau_bounded_nonuni is not None:
            num_final_eq = self.sim.delta_tau_unbounded_nonuni.shape[0]
            self.num_eq += num_final_eq

        # get euler pole kernel and rescale to match range of thetas;
        # it needs a utm zone, so only if the euler pole is estimated
        if "euler_pole" in self.psets_list:
            self.G_ep = self.sim.get_euler_pole_kernel() / 1e9
        # get time vector
        self.dt = self.sim.t_obs - self.sim.t_obs[0]

        # allocate memory
        ticks.append(self.sync_and_time())
        patches = self.fault.inner_num_patches
        SEC_PER_YEAR = 86400 * 365.25
        self.t_obs_sec = altar.cuda.vector(
            source=(self.sim.t_obs * SEC_PER_YEAR).astype(
                self.gpuprec, order="C", copy=False
            )
        )
        ievents = (
            [self.sim.ix_break_joint[-2]]
            + [i for i in self.sim.ix_eq_joint if i > self.sim.ix_break_joint[-2]]
            + [self.sim.ix_break_joint[-1]]
        )
        tevents = self.sim.t_eval_joint[ievents] - self.sim.t_eval_joint[ievents[0]]
        self.t_events = altar.cuda.vector(
            source=(tevents * SEC_PER_YEAR).astype(self.gpuprec, order="C", copy=False)
        )
        self.i_slips_obs = altar.cuda.vector(
            source=np.array([0] + self.sim.i_slips_obs).astype("int32")
        )
        self.K_inner_inner_onfault = altar.cuda.vector(
            source=np.ascontiguousarray(
                self.fault.K_inner_inner[:, :2, :, :2], dtype=self.gpuprec
            ).ravel()
        )
        K_inner_asperities_v_plate = self.sim.K_inner_asperities_v_plate.T.ravel()
        self.K_inner_asperities_v_plate = altar.cuda.vector(
            source=np.ascontiguousarray(K_inner_asperities_v_plate, dtype=self.gpuprec)
        )
        v_plate_ddcs_proj_eff_inner = self.sim.v_plate_ddcs_proj_eff[
            self.fault.s_inner, :
        ].T.ravel()
        self.v_plate_ddcs_proj_eff_inner = altar.cuda.vector(
            source=np.ascontiguousarray(v_plate_ddcs_proj_eff_inner, dtype=self.gpuprec)
        )

        # several copies of the initial state, one per system; the rate-dependent model
        # keeps it, and starts each batch from the end state of the previous one; the
        # traction-dependent one depends on alpha_h, so it is refilled for each batch
        state_init_arr = np.concatenate(
            [np.zeros(2 * patches), np.log(self.sim.v_init / self.rheo.v_0).T.ravel()]
        )
        self.state_init = altar.cuda.matrix(
            source=np.tile(state_init_arr, (self.cuda_batch_size, 1)).astype(
                self.gpuprec
            )
        )

        G_surf = (
            self.sim.G_surf[:, :, self.sim.fault.s_inner, :]
            .transpose(3, 2, 1, 0)
            .reshape(2 * patches, 3 * self.sim.n_observers)
        )
        self.G_surf = altar.cuda.matrix(
            source=np.ascontiguousarray(G_surf, dtype=self.gpuprec)
        )

        # simulate state - yeval in ode
        self.sim_state = altar.cuda.vector(
            shape=self.cuda_batch_size * self.sim.t_obs.size * patches * 4,
            dtype=self.gpuprec,
        )

        # copy the observation mask onto the GPU, as bytes
        mask = self.dataobs.mask
        if mask is None:
            self.obs_mask = altar.cuda.vector(
                source=np.ones(
                    self.sim.t_obs.size * 3 * self.sim.n_observers, dtype="uint8"
                )
            )
            channel.log(f"Device {self.device.id}: Assuming no masked values")
        else:
            self.obs_mask = altar.cuda.vector(source=mask.astype("uint8"))
            channel.log(
                f"Device {self.device.id}: Loaded mask with "
                f"{(~mask).sum()} masked values"
            )

        # load reference stations, if present
        if self.ref_station_indices is not None:
            self.i_stat_ref = altar.cuda.vector(
                source=np.asarray(self.ref_station_indices, dtype="int32")
            )
            self.n_stat_ref = len(self.ref_station_indices)
            channel.log(
                f"Device {self.device.id}: Found {self.n_stat_ref} reference stations"
            )
        else:
            self.i_stat_ref = altar.cuda.vector(source=np.asarray([], dtype="int32"))
            self.n_stat_ref = 0
            channel.log(f"Device {self.device.id}: No reference stations used")

        # predicted observations, and their likelihoods
        self.obs_disp = altar.cuda.matrix(
            shape=(
                self.cuda_batch_size,
                self.sim.t_obs.size * 3 * self.sim.n_observers,
            ),
            dtype=self.gpuprec,
        )
        self.likelihood_batch = altar.cuda.vector(
            shape=self.cuda_batch_size, dtype=self.gpuprec
        )

        # the per-batch inputs of the forward model
        self.alpha_h = altar.cuda.vector(
            shape=self.cuda_batch_size * patches, dtype=self.gpuprec
        )
        self.delta_tau_div_alpha_h = altar.cuda.vector(
            shape=self.cuda_batch_size * self.num_eq * patches * 2, dtype=self.gpuprec
        )
        self.obs_ref = altar.cuda.matrix(
            shape=(self.cuda_batch_size, self.sim.t_obs.size * 3), dtype=self.gpuprec
        )
        self.obs_ep = altar.cuda.vector(
            shape=self.cuda_batch_size * self.sim.n_observers * 2 * self.dt.size,
            dtype=self.gpuprec,
        )

        # create CUDA model for all samples (need to reduce to batch size)
        ticks.append(self.sync_and_time())
        precision = {"float64": "double", "float32": "float"}[self.gpuprec]
        method = "" if self.integrator == "dopri5" else f"_{self.integrator}"
        self.cmodel = getattr(
            getattr(libcudaseas, self.forward_ode), f"model_{precision}{method}"
        )()
        channel.log(
            f"Device {self.device.id}: Running {self.forward_ode} model with {self.integrator} "
            f"in {self.gpuprec} precision"
        )
        # the arguments the two integrators share
        args = dict(
            cuda_batch_size=self.cuda_batch_size,
            max_cycles=self.sim.n_cycles_max,
            num_t_obs=self.sim.t_obs.size,
            t_obs_sec=self.t_obs_sec.grid,
            num_ix_eq=self.sim.n_slips,
            num_eq=self.num_eq,
            t_events=self.t_events.grid,
            i_slips_obs=self.i_slips_obs.grid,
            n_slips_obs=self.sim.n_slips_obs + 1,
            v_0=self.rheo.v_0,
            mu_over_2vs=self.fault.mu_over_2vs,
            num_inner_patches=patches,
            K_inner_inner_onfault=self.K_inner_inner_onfault.grid,
            K_inner_asperities_v_plate=self.K_inner_asperities_v_plate.grid,
            v_plate_ddcs_proj_eff_inner=self.v_plate_ddcs_proj_eff_inner.grid,
            sim_state=self.sim_state.grid,
            atol=self.sim.atol,
            rtol=self.sim.rtol,
            spinup_atol=self.sim.spinup_atol,
            spinup_rtol=self.sim.spinup_rtol,
            num_stations=self.sim.n_observers,
            obs_mask=self.obs_mask.grid,
            i_stat_ref=self.i_stat_ref.grid,
            n_stat_ref=self.n_stat_ref,
            ref_vel_index=self.velocity_reference_index,
        )
        if self.forward_ode == "ratedependent":
            self.cmodel.initialize(**args, state_init=self.state_init.grid)
        else:  # self.forward_ode == "tractiondependent"
            self.cmodel.initialize(**args, rho=self.sim.rho)

        # remove precomputed farfield effects on observations
        ticks.append(self.sync_and_time())
        if np.any(self.sim.T_rec_logsigma) or (not self.sim.enforce_v_plate):
            raise NotImplementedError
        surf_disps_outer = get_surface_displacements(
            self.sim.outer_creep_slip, self.sim.G_surf[:, :, self.fault.s_outer, :]
        )
        surf_disps_lower = get_surface_displacements(
            self.sim.lower_creep_slip, self.sim.G_surf[:, :, self.fault.s_lower, :]
        )
        obs_farfield = (
            surf_disps_outer + surf_disps_lower
        ).T  # to change into CUDA ordering
        self.obs_farfield = altar.cuda.vector(
            source=obs_farfield.ravel().astype(dtype=self.gpuprec, order="C")
        )

        # initialize forward model variables on GPU
        ticks.append(self.sync_and_time())
        self.delta_tau_bounded_indices = altar.cuda.vector(
            source=self.sim.delta_tau_bounded_indices.astype("int32")
        )
        if self.sim.delta_tau_bounded_nonuni is None:
            self.delta_tau_bounded_indices_final = self.delta_tau_bounded_indices
        else:
            delta_tau_bounded_indices_final = self.sim.delta_tau_bounded_indices.copy()
            delta_tau_bounded_indices_final[-num_final_eq:] = np.arange(
                num_orig_eq, self.num_eq
            )
            self.delta_tau_bounded_indices_final = altar.cuda.vector(
                source=delta_tau_bounded_indices_final.astype("int32")
            )

        # print timings
        ticks.append(self.sync_and_time())
        channel.log(
            f"Device {self.device.id}: "
            f"Initialized SEAS3D in {ticks[-1] - ticks[0]:.1f}s"
        )

        # done
        return self

    def parse_theta(self, theta_arr):
        """
        Create a new Rheology object from a NumPy theta array and optionally
        return the rotation vector if contained in the parameter set.
        """
        # loop over theta entries
        rotvec = None
        rheo_kw_args = deepcopy(self.rheo_dict)
        if self.psets_list == ["log10_alpha_h_mat"]:  # single pset for entire matrix
            rheo_kw_args["alpha_h_mat"][self.theta_subset_indices] = (
                10 ** theta_arr.reshape(self.n_estim_row, -1)
            )
        else:  # parse individual rows of theta
            psets_list_alphah = [
                p for p in self.psets_list if p.startswith("log10_alpha_h_")
            ]
            n_theta_rows = len(psets_list_alphah)
            n_theta_cols = list(
                set([self.psets[name].count for name in psets_list_alphah])
            )
            assert len(n_theta_cols) == 1, "Different lengths of psets."
            n_theta_cols = n_theta_cols[0]
            assert (
                (n_theta_rows, n_theta_cols) == (self.n_estim_row, self.n_estim_col)
            ) or ((n_theta_rows, n_theta_cols) == (self.n_estim_row, 1)), (
                f"Expected theta of shape {(self.n_estim_row, self.n_estim_col)} (or "
                f"columns broadcastable), got shape {(n_theta_rows, n_theta_cols)}."
            )
            theta_out = np.full((n_theta_rows, n_theta_cols), np.nan)
            j = 0
            for name in self.psets_list:
                if name.startswith("log10_alpha_h_"):
                    irow = int(name[14:])
                    theta_out[irow, :] = theta_arr[j : j + n_theta_cols]
                elif name == "euler_pole":
                    rotvec = theta_arr[j : j + 3]
                j += self.psets[name].count
            rheo_kw_args["alpha_h_mat"][self.theta_subset_indices] = 10**theta_out
        # return new object instance
        return RateStateSteadyLogarithmic2D(**rheo_kw_args), rotvec

    def forward_model_batched(self, theta, prediction, batch=None):
        """
        SEAS forward model for the first {batch} samples
        :param theta: array (batch, parameters), physical parameters, on the host
        :param prediction: matrix (cuda_batch_size, observations), filled with the
                           predicted data of the first {batch} rows
        :param batch: integer, the number of samples to be computed,
                      batch<=cuda_batch_size
        :return: prediction as predicted data
        """

        # print info
        channel = self.info
        batch_size_run = theta.shape[0] if batch is None else batch
        patches = self.fault.inner_num_patches

        # create new rheology instances
        ticks = []
        ticks.append(self.sync_and_time())
        parsed_thetas = [
            self.parse_theta(np.array(theta[i], dtype=float))
            for i in range(batch_size_run)
        ]
        rheos = [pt[0] for pt in parsed_thetas]

        # create new simulation instances, modifying copied objects
        ticks.append(self.sync_and_time())
        sims = self.sims_storage[:batch_size_run]
        for i, sim in enumerate(sims):
            sim.update_rheo(rheos[i])

        # create stacked versions of alpha_h and delta_tau_div_alpha
        ticks.append(self.sync_and_time())
        alpha_h_vec_stacked = np.stack([s.alpha_h_vec.squeeze() for s in sims])
        np.asarray(self.alpha_h)[
            : batch_size_run * patches
        ] = alpha_h_vec_stacked.ravel()
        if self.sim.delta_tau_bounded_nonuni is None:
            dtau_bound_comp_list = [s.delta_tau_bounded_compressed for s in sims]
        else:
            dtau_bound_comp_list = [
                np.concatenate(
                    [s.delta_tau_bounded_compressed, s.delta_tau_bounded_nonuni], axis=0
                )
                for s in sims
            ]
        delta_tau_bounded_compressed_stacked = np.stack(dtau_bound_comp_list)
        # the traction integrator takes the stress changes as they are
        if self.forward_ode == "ratedependent":
            delta_tau_bound_comp_div_alpha = (
                delta_tau_bounded_compressed_stacked
                / alpha_h_vec_stacked[:, None, :, None]
            )
        else:  # self.forward_ode == "tractiondependent"
            delta_tau_bound_comp_div_alpha = delta_tau_bounded_compressed_stacked
        delta_tau_bound_comp_div_alpha = np.concatenate(
            [
                delta_tau_bound_comp_div_alpha[:, :, :, 0],
                delta_tau_bound_comp_div_alpha[:, :, :, 1],
            ],
            axis=2,
        )
        np.asarray(self.delta_tau_div_alpha_h)[
            : delta_tau_bound_comp_div_alpha.size
        ] = delta_tau_bound_comp_div_alpha.ravel()

        # calculate euler pole motion
        if "euler_pole" in self.psets_list:
            obs_ep_np = (
                np.stack(
                    [
                        self.G_ep.reshape(-1, 3) @ (pt[1][:, None] * self.dt[None, :])
                        for pt in parsed_thetas
                    ],
                    axis=0,
                )
                .reshape(batch_size_run, self.sim.n_observers, 2, self.dt.size)
                .transpose(0, 3, 2, 1)
            )
        else:
            obs_ep_np = np.zeros(
                (batch_size_run, self.dt.size, 2, self.sim.n_observers)
            )
        np.asarray(self.obs_ep)[: obs_ep_np.size] = obs_ep_np.ravel()

        # the inputs both integrators share
        args = dict(
            alpha_h_vec=self.alpha_h.grid,
            delta_tau_div_alpha_h=self.delta_tau_div_alpha_h.grid,
            delta_tau_bounded_indices=self.delta_tau_bounded_indices.grid,
            delta_tau_bounded_indices_final=self.delta_tau_bounded_indices_final.grid,
            G_surf=self.G_surf.grid,
            obs_disp=prediction.grid,
            ref_obs=self.obs_ref.grid,
            obs_farfield=self.obs_farfield.grid,
            obs_ep=self.obs_ep.grid,
            batches=batch_size_run,
            v_ratio_max=self.v_ratio_max,
            num_threads=self.cuda_threads,
            verbose=self.verbose,
        )

        if self.forward_ode == "ratedependent":

            # call CUDA forward model
            ticks.append(self.sync_and_time())
            self.cmodel.forward_model_batch(**args)

        else:  # self.forward_ode == "tractiondependent"

            # create state_init if integrating traction
            # initial patch state, (cuda_batch_size, 4 * num_inner_patches) [m|m|Pa|Pa]
            # tau_0 + alpha_h_vec * np.log(v / v_0) + mu_over_2vs * v

            # alpha_h_vec_stacked is [batch_size_run, patches]
            v_init_norm = np.linalg.norm(self.sim.v_init, axis=1)  # is [patches]
            tau_init_norm = (
                self.sim.rho + np.log(v_init_norm[None, :] / self.rheo.v_0)
            ) * alpha_h_vec_stacked + self.fault.mu_over_2vs * v_init_norm[None, :]
            # tau_init_norm  is also now [batch_size_run, patches]
            tau_init = self.sim.v_init[None, :, :] * (
                tau_init_norm[:, :, None] / v_init_norm[None, :, None]
            )  # is now [batch_size_run, patches, 2]

            # zero slip, and the traction, for each system in the batch
            state_init = np.asarray(self.state_init)
            state_init[:batch_size_run, : 2 * patches] = 0
            state_init[:batch_size_run, 2 * patches :] = tau_init.transpose(
                0, 2, 1
            ).reshape(batch_size_run, -1)

            # call CUDA forward model
            ticks.append(self.sync_and_time())
            self.cmodel.forward_model_batch(state_init=self.state_init.grid, **args)

        # log timings
        ticks.append(self.sync_and_time())
        infostr = (
            f"Device {self.device.id}: Ran forward_model_batched for {batch_size_run} "
            f"samples in {ticks[-1] - ticks[0]:.1f}s ("
        )
        infostr += (
            f"Rheology instances = {ticks[1] - ticks[0]:.1f}s, "
            f"Simulation instances = {ticks[2] - ticks[1]:.1f}s, "
            f"Stacked parameters = {ticks[3] - ticks[2]:.1f}s, "
        )
        infostr += f"CUDA forward model = {ticks[4] - ticks[3]:.1f}s)"
        channel.log(infostr)
        # the integrator's step statistics, summed over the batch
        if self.verbose:
            stats = {k: np.asarray(v) for k, v in self.cmodel.step_statistics().items()}
            channel.log(
                f"Device {self.device.id}: steps accepted {stats['accepted'].sum()}, "
                f"rejected {stats['rejected'].sum()}, "
                f"stability limited {stats['stiff'].sum()}; "
                f"spin-up cycles {stats['cycles'].min()}-{stats['cycles'].max()}; "
                f"smallest step {stats['hmin'].min():.3g}s"
            )

        # all done
        return prediction

    def eval_data_likelihood(self, theta, likelihood, batch=None):
        """
        Compute the likelihood from my forward problem, in batches of {cuda_batch_size}
        :param: theta - physical parameters, matrix of (samples, parameters)
        :param: likelihood - computed likelihood, vector of (samples)
        :param: batch - number of samples to be computed
        """
        batch = theta.shape[0] if batch is None else batch
        # theta and the likelihood live in managed memory, so the host can see them
        theta = np.asarray(theta)
        llk = np.asarray(likelihood)

        # the data storage for data prediction and its likelihood
        predictions = self.obs_disp
        llk_batch = self.likelihood_batch

        # iterate over batches
        cuda_batch_size = self.cuda_batch_size
        for system_start in range(0, batch, cuda_batch_size):
            # get the actual batch size
            batch_size_run = min(cuda_batch_size, batch - system_start)
            # call forward model to calculate the data prediction
            self.forward_model_batched(
                theta=theta[system_start : system_start + batch_size_run],
                prediction=predictions,
                batch=batch_size_run,
            )
            # the l2 norm of its residual, applying the data covariance to it
            self.dataobs.eval_likelihood(
                prediction=predictions,
                likelihood=llk_batch,
                residual=False,
                whitened=False,
                batch=batch_size_run,
            )
            # copy likelihood to global
            llk[system_start : system_start + batch_size_run] = np.asarray(llk_batch)[
                :batch_size_run
            ]
            # the samples whose integration failed are as unlikely as can be
            self.reject_failed(llk[system_start : system_start + batch_size_run])

        # all done
        return self

    @altar.export
    def gradient(self, controller, step, batch=None):
        """
        Fill {step.prior_gradient} and {step.data_gradient} with the gradients of the log prior
        and of the data log likelihood w.r.t. the physical parameters, the latter by central
        differences of the forward model, all samples and perturbations batched together
        """
        # the differences need the likelihood to many more digits than float32 keeps
        if self.precision != "float64":
            self.error.log("the seas3d gradient needs job.gpuprecision = float64")
            raise SystemExit(1)
        # in an ensemble, the ensemble owns the parameter sets and their priors
        if not self.embedded and not self.checked_unbounded_priors:
            self.verify_unbounded_priors()

        theta = self.restrict(theta=step.theta)
        batch = theta.shape[0] if batch is None else batch
        grad_prior = self.restrict(theta=step.prior_gradient)
        for name in ([] if self.embedded else self.psets_list):
            self.psets[name].prior_gradient(theta=theta, gradient=grad_prior, batch=batch)

        # each sample, shifted by +h and -h along each parameter
        x = np.array(np.asarray(theta)[:batch], dtype=float)
        parameters = x.shape[1]
        h = self.gradient_step
        shifted = np.repeat(x[:, None, :], 2 * parameters, axis=1)
        for j in range(parameters):
            shifted[:, 2 * j, j] += h
            shifted[:, 2 * j + 1, j] -= h
        n = batch * 2 * parameters
        theta_fd = altar.cuda.matrix(source=shifted.reshape(n, parameters), dtype=self.precision)
        llk_fd = altar.cuda.vector(shape=n, dtype=self.precision)
        self.eval_data_likelihood(theta=theta_fd, likelihood=llk_fd, batch=n)

        # the central differences; a sample with a failed integration gets no gradient
        llk = np.asarray(llk_fd).reshape(batch, parameters, 2)
        grad = (llk[:, :, 0] - llk[:, :, 1]) / (2 * h)
        grad[(llk == np.finfo(llk.dtype).min).any(axis=(1, 2))] = 0
        np.asarray(self.restrict(theta=step.data_gradient))[:batch] = grad
        # all done
        return self

    def reject_failed(self, llk):
        """
        Give the samples of the last batch whose integration failed the lowest
        likelihood, finite so that the annealing weights stay well defined
        """
        failed = np.asarray(self.cmodel.step_statistics()["failed"], dtype=bool)[
            : llk.size
        ]
        if failed.any():
            llk[failed] = np.finfo(llk.dtype).min
            self.info.log(
                f"Device {self.device.id}: {failed.sum()} of {failed.size} samples "
                f"failed to integrate, and are rejected"
            )
        return self


# end of file
