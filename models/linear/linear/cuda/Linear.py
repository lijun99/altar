# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import numpy
# the package
import altar
import altar.cuda


# declaration
class Linear:
    """
    The cuda implementation of the linear forward model: data = G theta

    {G} is pre-merged with the data covariance's inverse Cholesky factor once, at
    {initialize} (see {_premerge_covariance}) -- {altar.data.cuda.DataL2} always pre-merges
    the same factor into {dataobs_batch} regardless of {merge_cd_with_data} (unlike cpu,
    which applies it lazily per {gradient} call), so premerging {G} is required for
    correctness here, not an optional speedup: every {forward_model_batched} call then
    produces an already-whitened residual, matching {DataL2.eval_likelihood}'s contract of
    never applying {sigma_inv} itself.
    """


    def initialize(self, model, application):
        """
        Load the Green functions, upload them, and pre-merge the data covariance into them
        """
        self.precision = model.precision
        self.observations = model.observations
        self.parameters = model.parameters
        # {eval_likelihood}/{dataobs_batch} live on {model.dataobs}; kept for
        # {forward_model_batched}, which doesn't otherwise receive {model}
        self._dataobs = model.dataobs

        # {model.io.load} always returns a cpu gsl object regardless of backend; upload it
        G_cpu = model.io.load(filename=model.green, shape=(self.observations, self.parameters))
        self.G_host = numpy.array(G_cpu, dtype=float)
        self.G = altar.cuda.matrix(source=G_cpu, dtype=self.precision)

        # pre-merge the covariance's Cholesky factor into G, once
        self._premerge_covariance()
        # all done
        return self


    def _premerge_covariance(self):
        """
        G <- U @ G, in place; U is {self._dataobs.cd_inv}'s row-major-upper Cholesky factor
        (Cd_inv = U^T U). The matrix generalization of
        {altar.data.cuda.DataL2.merge_cdto_data}'s vector premerge.

        {cublas.Operation.N} is correct here (not {.T}): {cd_inv}'s raw cublas reading is
        already U^T (its row-major-upper storage read column-major), and the target
        col-major result (G' with swapped dims, i.e. G'^T = G^T @ U^T) needs op(A) = U^T
        directly, so op=N. Contrast {merge_cdto_data}'s trmv, which needs op=T -- there the
        target is `U @ vector` directly, with no row-major/col-major dimension swap in play
        since a vector has no shape ambiguity. Verified against a real GPU run (comparing
        against a plain numpy {U @ G}) before being written here.
        """
        cd_inv = self._dataobs.cd_inv
        # a constant variance: U is the scalar cd_inv; a one-off scaling
        if isinstance(cd_inv, float):
            numpy.asarray(self.G)[:, :] *= cd_inv
            return self
        cublas = altar.cuda.cublas
        handle = altar.cuda.cublas_handle()
        obs, par = self.observations, self.parameters
        trmm = cublas.dtrmm if self.precision == "float64" else cublas.strmm
        trmm(
            handle,
            cublas.SideMode.RIGHT, cublas.FillMode.LOWER, cublas.Operation.N,
            cublas.DiagType.NON_UNIT,
            par, obs, 1.0,
            cd_inv.grid if hasattr(cd_inv, "grid") else cd_inv, obs,
            self.G.grid, par,
            self.G.grid, par,
        )
        return self


    def forward_model_batched(self, model, theta, prediction, batch=None):
        """
        Fill {prediction}, shape (samples x observations), with the residual G'·θ - d' for
        each sample in {theta} -- already whitened, since both G and the observed data
        ({self._dataobs.dataobs_batch}) are pre-merged with the same covariance factor.

        The gemm below computes this directly in the (samples x observations) row-major
        layout cublas needs no separate transpose step for, unlike the cpu path (which
        computes in (observations x samples) then calls {.transpose()}): declaring G/theta
        with their dimensions swapped reads them as their own transpose (the same row-major/
        col-major trick used throughout this codebase's cuda layer), and this particular
        combination of operand roles and transpose flags happens to land the result directly
        in the desired row-major layout with no extra transpose needed. Verified against a
        real GPU run (comparing against plain numpy {theta @ G.T - dataobs_batch}) before
        being written here.
        """
        cublas = altar.cuda.cublas
        handle = altar.cuda.cublas_handle()
        samples = batch if batch is not None else theta.shape[0]
        gemm = cublas.dgemm if self.precision == "float64" else cublas.sgemm

        prediction.copy(self._dataobs.dataobs_batch)
        gemm(
            handle,
            cublas.Operation.T, cublas.Operation.N,
            self.observations, samples, self.parameters,
            1.0,
            self.G.grid, self.parameters,
            theta.grid, self.parameters,
            -1.0,
            prediction.grid, self.observations,
        )
        # all done
        return self


    def forward_model(self, model, theta, green=None, prediction=None, observation=None):
        """
        Linear forward model prediction = G * theta for a single sample; not used by any
        cuda code path today ({forward_problem} works on the host), so it is not yet
        implemented here
        """
        raise NotImplementedError(
            "cuda 'Linear.forward_model' (single-sample) is not implemented; "
            "use 'forward_model_batched' instead")


    def covariance_updated(self, model):
        """
        Premerge the new covariance into a fresh copy of the raw green's functions
        """
        numpy.asarray(self.G)[:, :] = self.G_host
        self._premerge_covariance()
        return self


    def green(self):
        """
        The raw green's functions (observations x parameters), as a numpy array
        """
        return self.G_host


    def gradient(self, model, controller, step, batch=None):
        """
        Fill {step.prior_gradient} and {step.data_gradient} with the gradients of the log
        prior and log data likelihood with respect to {step.theta}, for use by gradient-based
        samplers (e.g. SGLD, HMC) -- the naming cuda state objects use (see
        {altar.bayesian.states.cuda.HMCState}/{LangevinState}/{LangevinStep} and
        {CUDASGLD.estimate_rate}'s own {step.data_gradient}/{step.prior_gradient} reads),
        unlike their cpu counterparts' {grad_prior}/{grad_data}.

        grad_data = -G'^T @ w, where w is the (already-whitened) residual from
        {forward_model_batched}. Algebraically this is cpu's 3-step chain
        (r = Gθ-d; w = Lr; wt = L^T w; grad = -G^T wt) collapsed to a single gemm, since G'
        is already premerged: G'^T = (LG)^T = G^T L^T, so -G'^T w = -G^T L^T (L r) = -G^T
        L^T wt, matching the cpu formula exactly.
        """
        if not model.checked_unbounded_priors:
            model.verify_unbounded_priors()

        θ = model.restrict(theta=step.theta)
        grad_prior = model.restrict(theta=step.prior_gradient)
        grad_data = model.restrict(theta=step.data_gradient)
        for name in model.psets_list:
            model.psets[name].prior_gradient(theta=θ, gradient=grad_prior, batch=batch)

        samples = batch if batch is not None else θ.shape[0]
        w = altar.cuda.matrix(shape=(samples, self.observations), dtype=self.precision)
        self.forward_model_batched(model=model, theta=θ, prediction=w, batch=batch)

        cublas = altar.cuda.cublas
        handle = altar.cuda.cublas_handle()
        gemm = cublas.dgemm if self.precision == "float64" else cublas.sgemm
        gemm(
            handle,
            cublas.Operation.N, cublas.Operation.N,
            self.parameters, samples, self.observations,
            -1.0,
            self.G.grid, self.parameters,
            w.grid, self.observations,
            0.0,
            grad_data.grid, self.parameters,
        )
        # all done
        return self


    # private data
    G = None # the (premerged) Green functions
    G_host = None # the raw Green functions, on the host, for {forward_problem}
    observations = None
    parameters = None
    precision = None
    _dataobs = None # {model.dataobs}, for {dataobs_batch} in {forward_model_batched}


# end of file
