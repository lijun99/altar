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
from seqeas.subduction3d import RateStateSteadyLogarithmic2D, Fault3D, SubductionSimulation3D


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
    v_init_file = altar.properties.str(default=None)
    cuda_threads = altar.properties.int(default=0)
    verbose = altar.properties.bool(default=False)
    alpha_h_mat_rows = altar.properties.int()
    mask_file = altar.properties.path(default=None)

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
        self.rheo = RateStateSteadyLogarithmic2D(**self.rheo_dict)
        self.fault = Fault3D(**self.fault_dict)
        self.sim = SubductionSimulation3D(**self.sim_dict, rheo=self.rheo, fault=self.fault)

        # load initial velocity
        if self.v_init_file is not None:
            try:
                v_init_loaded = np.load(self.v_init_file)
                exp_shape = self.sim.v_plate_ddcs_proj_eff[self.sim.fault.s_inner, :].shape
                assert v_init_loaded.shape == exp_shape
            except AssertionError as e:
                raise AssertionError("Couldn't load initial velocities due to shape mismatch:\n"
                                     f"Loaded = {v_init_loaded.shape}, expected = {exp_shape}."
                                     ).with_traceback(e.__traceback__) from e
            except FileNotFoundError as e:
                raise FileNotFoundError("Couldn't find initial velocities file "
                                        f"'{self.v_init_file}'."
                                        ).with_traceback(e.__traceback__) from e
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
            shape=self.cuda_batch_size * self.sim.t_obs.size * self.fault.inner_num_patches * 4,
            dtype=self.gpuprec)

        # predicted observations
        self.obs_disp = altar.cuda.matrix(
            shape=(self.cuda_batch_size, self.sim.t_obs.size * 3 * self.sim.n_observers),
            dtype=self.gpuprec)

        # mask/weight for observations
        if self.mask_file is None:
            self.mask = None
            # or create a unit vector
            # self.mask = altar.cuda.vector(shape=self.dataobs.gDataVec.shape,
            #                               dtype=self.gpuprec).fill(1)
        else:
            # please implement this - to read mask of data from a file
            # TODO
            self.mask = None

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
            self.cuda_batch_size,
            self.sim.n_cycles_max,
            self.sim.t_obs.size,
            self.t_obs_sec.data,
            self.sim.n_slips,
            self.sim.delta_tau_bounded_compressed.shape[0],
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

        # TODO: this is still necessary!!!
        # # remove precomputed farfield effects on observations
        # ticks.append(self.sync_and_time())
        # if np.any(self.sim.T_rec_logsigma) or (not self.sim.enforce_v_plate):
        #     raise NotImplementedError
        # surf_disps_outer = get_surface_displacements(
        #     self.sim.outer_creep_slip, self.sim.G_surf[:, :, self.fault.s_outer, :])
        # surf_disps_lower = get_surface_displacements(
        #     self.sim.lower_creep_slip, self.sim.G_surf[:, :, self.fault.s_lower, :])
        # self.dataobs.dataobs[:] -= self.sim.zero_obs_at_eq(surf_disps_outer + surf_disps_lower
        #                                                    ).T.ravel()
        # # after any change of dataobs, update to cuda objects is needed
        # self.dataobs.updateCovariance()

        # initialize forward model variables on GPU
        ticks.append(self.sync_and_time())
        self.alpha_h = altar.cuda.vector(
            shape=self.cuda_batch_size * self.fault.inner_num_patches, dtype=self.gpuprec)
        self.delta_tau_div_alpha_h = altar.cuda.vector(
            shape=(self.cuda_batch_size * self.sim.delta_tau_bounded_compressed.shape[0]
                   * self.fault.inner_num_patches * 2),
            dtype=self.gpuprec)

        # print timings
        ticks.append(self.sync_and_time())
        channel.log(f"Initialized SEAS3D in {ticks[-1] - ticks[0]:.1f}s")
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
        assert self.psets_list == ["log10_alpha_h_mat"]
        rheo_kw_args["alpha_h_mat"] = \
            10**np.array(theta_arr).reshape(self.alpha_h_mat_rows, -1)
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
        channel.log("Time between forwardModelBatched calls: "
                    f"{self.sync_and_time() - self.timer_fmb}s")

        # create new rheology instances
        ticks = []
        ticks.append(self.sync_and_time())
        rheos = [self.rheo_from_theta(theta.get_row(i).copy_to_host(type="numpy"))
                 for i in range(batch_size_run)]

        # create new simulation instances, reusing G_surf
        ticks.append(self.sync_and_time())
        sims = [SubductionSimulation3D(**self.sim_dict, rheo=rheos[i],
                                       fault=self.fault, G_surf=self.sim.G_surf)
                for i in range(batch_size_run)]

        # create stacked versions of alpha_h and delta_tau_div_alpha
        ticks.append(self.sync_and_time())
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
            source=np.ascontiguousarray(delta_tau_bound_comp_div_alpha,
                                        dtype=self.gpuprec))

        # call CUDA forward model
        ticks.append(self.sync_and_time())
        self.cmodel.forward_model_batch(self.alpha_h.data,
                                        self.delta_tau_div_alpha_h.data,
                                        self.G_surf.data,
                                        prediction.data,
                                        batch_size_run,
                                        self.cuda_threads,
                                        self.verbose)

        # TODO subsample the CUDA forward model for each station according to its availability

        # log timings
        ticks.append(self.sync_and_time())
        infostr = (f"Ran forwardModelBatched for {batch_size_run} samples in "
                   f"{ticks[-1] - ticks[0]:.1f}s (")
        infostr += (f"Rheology instances = {ticks[1] - ticks[0]:.1f}s, "
                    f"Simulation instances = {ticks[2] - ticks[1]:.1f}s, "
                    f"Stacked parameters = {ticks[3] - ticks[2]:.1f}s, ")
        infostr += f"CUDA forward model = {ticks[4] - ticks[3]:.1f}s)"
        channel.log(infostr)

        # all done
        self.timer_fmb = self.sync_and_time()
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

        # get logger
        channel = self.info

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
            channel.log(f"cuEvalLikelihood loop processing systems {system_start} to "
                        f"{system_start + batch_size_run - 1}")
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

            # NOTE: prediction here is the surface displacement.
            # If you want to correct them in python, you may add code here.

            predictions.subtractVector(vector=data_obs, size=(batch_size_run, observations))
            # call data method to calculate the l2 norm
            self.dataobs.cuEvalLikelihood(prediction=predictions, likelihood=likelihood_batch,
                                          residual=True, batch=batch_size_run, weight=self.mask)
            # copy likelihood to global
            likelihood.copytile(likelihood_batch, start=system_start, size=batch_size_run)

        # consider cd_inv as a constant
        cd_inv = self.dataobs.gcd_inv
        if isinstance(cd_inv, float):
            likelihood *= cd_inv
        else:
            raise NotImplementedError

        # return the likelihood
        return likelihood
