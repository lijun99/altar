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
# my base class
from altar.models.BayesianL2 import BayesianL2


# declaration
class Kinematic(BayesianL2, family="altar.models.seismic.kinematic"):
    """
    The kinematic slip model: rupture fronts by fast sweeping from the hypocenter, slips of each
    patch over time, then data = Gb Mb

    On the cuda backend, or on the cpu, where the samples run in parallel threads. My parameters,
    in {idx_map} order, are the strike slips, dip slips, rise times and rupture velocities of the
    {Nas}x{Ndd} patches, then the hypocenter along strike and dip.
    """


    # user configurable state
    green = altar.properties.path(default="kinematicG.gf.h5")
    green.doc = "the big-G green's functions, (2*Nas*Ndd*Nt, observations)"

    Nas = altar.properties.int(default=1)
    Nas.doc = "number of patches along strike direction"

    Ndd = altar.properties.int(default=1)
    Ndd.doc = "number of patches along dip direction"

    Nmesh = altar.properties.int(default=1)
    Nmesh.doc = "number of mesh points for each patch for fast sweeping"

    dsp = altar.properties.float(default=10.0)
    dsp.doc = "the length of each patch, in km"

    Nt = altar.properties.int(default=1)
    Nt.doc = "number of time intervals for the kinematic process"

    Npt = altar.properties.int(default=1)
    Npt.doc = "number of mesh points for each time interval for fast sweeping"

    dt = altar.properties.float(default=1.0)
    dt.doc = "the length of each time interval, in s"

    t0s = altar.properties.array(default=None)
    t0s.doc = "the start time of each patch"

    # the inputs for estimating the model uncertainty C_p, with cp=altar.models.cp.adaptive
    cmu_file = altar.properties.path(default="kinematicG.Cmu.h5")
    cmu_file.doc = "C_mu, the covariance of the uncertain model inputs mu, (n x n)"

    kmu_file = altar.properties.path(default="kinematicG.kernel.h5")
    kmu_file.doc = "the sensitivity kernels dGb/dmu_i, one (2*Nas*Ndd*Nt, observations) dataset each"

    idx_map = altar.properties.list(schema=altar.properties.int())
    idx_map.default = None
    idx_map.doc = "the columns of theta holding my parameters; default all, in order"

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model given a {problem} specification
        """
        self.native = altar.backends.active() != "cuda"
        # chain up; mounts my inputs, loads the data and its covariance, lays out my psets
        super().initialize(application=application)

        patches = self.Nas * self.Ndd
        self.NGbparameters = 2 * patches * self.Nt

        # my columns of theta
        idx_map = numpy.arange(self.parameters) if self.idx_map is None else numpy.asarray(self.idx_map)
        if idx_map.size != 4 * patches + 2:
            application.error.log(
                f"the kinematic model needs {4 * patches + 2} parameters for {patches} patches, "
                f"got {idx_map.size}")
            raise SystemExit(1)
        t0s = numpy.zeros(patches) if self.t0s is None else numpy.asarray(self.t0s, dtype=float)
        # the green's functions, kept raw for the forward problem
        self.GF = self.io.load(filename=self.green, shape=(self.NGbparameters, self.observations))

        if self.native:
            from altar.models.seismic.ext import seismic
            self._native_green()
            self.cmodel = seismic.Kinematic(
                Nas=self.Nas, Ndd=self.Ndd, Nmesh=self.Nmesh, dsp=self.dsp, Nt=self.Nt, Npt=self.Npt,
                dt=self.dt, t0s=t0s, idx_map=idx_map.astype("int64"), precision=self.precision)
            return self

        from altar.models.seismic.ext import cudaseismic
        self.gidx_map = altar.cuda.vector(source=idx_map.astype("int64"))
        self.gt0s = altar.cuda.vector(source=t0s, dtype=self.precision)
        # premerged with the data covariance for sampling
        self.gGF = altar.cuda.matrix(source=self.GF, dtype=self.precision)
        self._premerge_covariance()

        self.cmodel = cudaseismic.cudaKinematic(
            Nas=self.Nas, Ndd=self.Ndd, Nmesh=self.Nmesh, dsp=self.dsp,
            Nt=self.Nt, Npt=self.Npt, dt=self.dt, t0s=self.gt0s.grid,
            samples=self.samples, parameters=self.parameters, observations=self.observations,
            idx_map=self.gidx_map.grid)
        # all done
        return self


    def _native_green(self):
        """
        The green's functions of the cpu likelihood, at my precision: whitened by the data
        covariance, G <- G L with cd^-1 = L L^T, if the data are
        """
        G = numpy.asarray(self.GF, dtype=float)
        if self.dataobs.merge_cd_with_data:
            cd_inv = self.dataobs.cd_inv
            G = G * cd_inv if isinstance(cd_inv, float) else G @ numpy.asarray(cd_inv, dtype=float)
        self._G = numpy.ascontiguousarray(G, dtype=self.precision)
        return self


    def _native_residual(self, theta, batch):
        """
        The residuals Gb Mb - d of the first {batch} rows of {theta}, whitened if the data are, on
        the cpu, and the slips Mb
        """
        θ = numpy.ascontiguousarray(theta[:batch], dtype=self.precision)
        if self._mb is None or self._mb.shape[0] < batch:
            self._mb = numpy.zeros((max(batch, self.samples), self.NGbparameters), dtype=self.precision)
        Mb = self._mb[:batch]
        self.cmodel.cast_mb(theta=θ, mb=Mb, batch=batch)
        return Mb @ self._G - self.dataobs.dataobs, θ


    def _premerge_covariance(self):
        """
        G <- U G, with Cd_inv = U^T U; {self.gGF} holds G^T row-major, i.e. G column-major, and
        {cd_inv} read column-major is U^T, hence LEFT/LOWER/T; see {altar.models.linear.cuda}
        """
        cd_inv = self.dataobs.cd_inv
        # a constant variance: U is the scalar cd_inv; a one-off scaling
        if isinstance(cd_inv, float):
            numpy.asarray(self.gGF)[:, :] *= cd_inv
            return self
        cublas = altar.cuda.cublas
        obs = self.observations
        trmm = cublas.dtrmm if self.precision == "float64" else cublas.strmm
        trmm(
            altar.cuda.cublas_handle(),
            cublas.SideMode.LEFT, cublas.FillMode.LOWER, cublas.Operation.T,
            cublas.DiagType.NON_UNIT,
            obs, self.NGbparameters, 1.0,
            cd_inv.grid if hasattr(cd_inv, "grid") else cd_inv, obs,
            self.gGF.grid, obs,
            self.gGF.grid, obs,
        )
        return self


    def compute_cp(self, theta):
        """
        C_p = K_p C_mu K_p^T for the mean model {theta}, K_p[:, i] = Mb(theta) K_i
        """
        from altar.models.cp import sensitivity
        Mb = self.forward_problem(application=None, theta=numpy.asarray(theta)[None, :])["Mb"][0]
        shape = (self.NGbparameters, self.observations)
        return sensitivity(model=self, cmu_file=self.cmu_file, kmu_file=self.kmu_file,
                           predict=lambda kernel: Mb @ kernel.reshape(shape))


    def covariance_updated(self):
        """
        Premerge the new covariance into a fresh copy of the raw green's functions, once they exist
        """
        if self.native:
            if self._G is not None:
                self._native_green()
        elif self.gGF is not None:
            numpy.asarray(self.gGF)[:, :] = numpy.asarray(self.GF)
            self._premerge_covariance()
        return self


    def gradient(self, controller, step, batch=None):
        """
        Fill {step.prior_gradient} and {step.data_gradient} for the gradient-based samplers: by the
        chain rule through the slip history Mb, d(log L)/d(Mb) = -Gb'^T w, then back through the
        source time functions, the arrival times and their fast sweeping, on the device
        """
        if self.native:
            return self._native_gradient(step=step, batch=batch)
        # in an ensemble, the ensemble owns the parameter sets and their priors
        if not self.embedded:
            if not self.checked_unbounded_priors:
                self.verify_unbounded_priors()
            for name in self.psets_list:
                self.psets[name].prior_gradient(theta=step.theta, gradient=step.prior_gradient, batch=batch)

        θ = step.theta
        rows = θ.shape[0]
        samples = rows if batch is None else batch
        # the whitened residual w at {θ}, and d(log L)/d(Mb) = -Gb'^T w, in buffers made once
        if self._residual is None:
            self._residual = altar.cuda.matrix(shape=(rows, self.observations), dtype=self.precision)
            self._dmb = altar.cuda.matrix(shape=(rows, self.NGbparameters), dtype=self.precision)
        w, gdMb = self._residual, self._dmb
        self.forward_model_batched(theta=θ, prediction=w, batch=samples)
        cublas = altar.cuda.cublas
        gemm = cublas.dgemm if self.precision == "float64" else cublas.sgemm
        gemm(altar.cuda.cublas_handle(), cublas.Operation.T, cublas.Operation.N,
             self.NGbparameters, samples, self.observations, -1.0,
             self.gGF.grid, self.observations, w.grid, self.observations,
             0.0, gdMb.grid, self.NGbparameters)
        # and back to my columns of theta
        step.data_gradient.zero()
        self.cmodel.gradient(theta=θ.grid, dmb=gdMb.grid, gradient=step.data_gradient.grid, batch=samples)
        return self


    def _native_gradient(self, step, batch=None):
        """
        The gradients of the log prior and log likelihood on the cpu, into {step.grad_prior} and
        {step.grad_data}: d(log L)/d(Mb) = -C^-1 r Gb^T, then back through the slip history
        """
        θ = self.restrict(theta=step.theta)
        grad_prior = self.restrict(theta=step.grad_prior)
        grad_data = self.restrict(theta=step.grad_data)
        # in an ensemble, the ensemble owns the parameter sets and their priors
        if not self.embedded:
            if not self.checked_unbounded_priors:
                self.verify_unbounded_priors()
            for name in self.psets_list:
                self.psets[name].prior_gradient(theta=θ, gradient=grad_prior)
        batch = θ.shape[0] if batch is None else batch
        r, θb = self._native_residual(theta=θ, batch=batch)
        # the residual weighted by the inverse covariance, unless whitened already
        data = self.dataobs
        if not data.merge_cd_with_data:
            cd_inv = data.cd_inv
            r = r * (cd_inv * cd_inv) if isinstance(cd_inv, float) else r @ cd_inv @ cd_inv.T
        if data.mask is not None:
            r[:, ~data.mask] = 0
        dMb = numpy.ascontiguousarray(-r @ self._G.T, dtype=self.precision)
        gradient = numpy.zeros_like(θb)
        self.cmodel.gradient(theta=θb, dmb=dMb, gradient=gradient, batch=batch)
        grad_data[:batch] = gradient
        return self


    def forward_model_batched(self, theta, prediction, batch=None):
        """
        Fill {prediction}, (samples x observations), with the residual Gb Mb - d, whitened on
        cuda, and on the cpu if the data are
        """
        batch = batch if batch is not None else theta.shape[0]
        if self.native:
            prediction[:batch] = self._native_residual(theta=theta, batch=batch)[0]
            return self
        prediction.copy(self.dataobs.dataobs_batch)
        self.cmodel.forward_batched(
            theta=theta.grid, green=self.gGF.grid, prediction=prediction.grid,
            batch=batch, residual=True)
        return self


    @altar.export
    def forward_problem(self, application, theta):
        """
        The raw predicted data and the slips of each patch over time, "Mb", for each row of
        {theta}, in chunks of my sample count; see {altar.models.Model}
        """
        theta = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        rows = theta.shape[0]
        if self.native:
            Mb = numpy.zeros((rows, self.NGbparameters), dtype=self.precision)
            self.cmodel.cast_mb(theta=numpy.ascontiguousarray(theta, dtype=self.precision), mb=Mb, batch=rows)
            Mb = Mb.astype(float)
            return {"data": Mb @ numpy.asarray(self.GF, dtype=float), "Mb": Mb}
        data = numpy.empty((rows, self.observations))
        Mb = numpy.empty((rows, self.NGbparameters))
        # the raw green's functions, not the covariance-premerged ones
        gGF = altar.cuda.matrix(source=self.GF, dtype=self.precision)
        gtheta = altar.cuda.matrix(shape=(self.samples, self.parameters), dtype=self.precision)
        gMb = altar.cuda.matrix(shape=(self.samples, self.NGbparameters), dtype=self.precision)
        gData = altar.cuda.matrix(shape=(self.samples, self.observations), dtype=self.precision)
        for start in range(0, rows, self.samples):
            batch = min(self.samples, rows - start)
            numpy.asarray(gtheta)[:batch] = theta[start:start + batch]
            self.cmodel.cast_mb(theta=gtheta.grid, mb=gMb.grid, batch=batch)
            self.cmodel.linear_gm(green=gGF.grid, mb=gMb.grid, prediction=gData.grid, batch=batch, residual=False)
            data[start:start + batch] = numpy.asarray(gData)[:batch]
            Mb[start:start + batch] = numpy.asarray(gMb)[:batch]
        return {"data": data, "Mb": Mb}


    # private data
    native = False # whether on the cpu
    GF = None # the raw green's functions, on the host
    _G = None # the cpu likelihood's green's functions, whitened if the data are
    _mb = None # the slips of the cpu likelihood
    gGF = None # the covariance-premerged green's functions
    gt0s = None
    gidx_map = None
    _residual = None # the whitened residual of {gradient}
    _dmb = None # d(log L)/d(Mb) of {gradient}
    cmodel = None
    NGbparameters = None


# end of file
