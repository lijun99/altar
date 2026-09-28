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

    cuda only: the fast sweeping is too expensive for the cpu backend. My parameters, in
    {idx_map} order, are the strike slips, dip slips, rise times and rupture velocities of the
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
        # cuda only
        if altar.backends.active() != "cuda":
            application.error.log(
                "the kinematic model has only a cuda implementation; run it with job.gpus >= 1")
            raise SystemExit(1)

        # chain up; mounts my inputs, loads the data and its covariance, lays out my psets
        super().initialize(application=application)

        from altar.models.seismic.ext import cudaseismic
        patches = self.Nas * self.Ndd
        self.NGbparameters = 2 * patches * self.Nt

        # my columns of theta
        idx_map = numpy.arange(self.parameters) if self.idx_map is None else numpy.asarray(self.idx_map)
        if idx_map.size != 4 * patches + 2:
            application.error.log(
                f"the kinematic model needs {4 * patches + 2} parameters for {patches} patches, "
                f"got {idx_map.size}")
            raise SystemExit(1)
        self.gidx_map = altar.cuda.vector(source=idx_map.astype("int64"))
        t0s = numpy.zeros(patches) if self.t0s is None else numpy.asarray(self.t0s, dtype=float)
        self.gt0s = altar.cuda.vector(source=t0s, dtype=self.precision)

        # the green's functions, kept raw for the forward problem, and premerged with the data
        # covariance for sampling
        self.GF = self.io.load(filename=self.green, shape=(self.NGbparameters, self.observations))
        self.gGF = altar.cuda.matrix(source=self.GF, dtype=self.precision)
        self._premerge_covariance()

        self.cmodel = cudaseismic.cudaKinematic(
            Nas=self.Nas, Ndd=self.Ndd, Nmesh=self.Nmesh, dsp=self.dsp,
            Nt=self.Nt, Npt=self.Npt, dt=self.dt, t0s=self.gt0s.grid,
            samples=self.samples, parameters=self.parameters, observations=self.observations,
            idx_map=self.gidx_map.grid)
        # all done
        return self


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
        if self.gGF is not None:
            numpy.asarray(self.gGF)[:, :] = numpy.asarray(self.GF)
            self._premerge_covariance()
        return self


    def forward_model_batched(self, theta, prediction, batch=None):
        """
        Fill {prediction}, (samples x observations), with the whitened residual Gb' Mb - d'
        """
        batch = batch if batch is not None else theta.shape[0]
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
    GF = None # the raw green's functions, on the host
    gGF = None # the covariance-premerged green's functions
    gt0s = None
    gidx_map = None
    cmodel = None
    NGbparameters = None


# end of file
