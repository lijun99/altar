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
class KinematicGaussianSlip(BayesianL2, family="altar.models.seismic.kinematicgaussianslip"):
    """
    The kinematic model on a triangulated fault, for static displacements binned in time, e.g.
    GNSS, with the slips integrated out under a gaussian prior: only the rupture fronts are sampled

    Rupture fronts from one or more nucleation points, each with its hypocenter, origin time and
    rupture velocity, by the fast iterative method; each patch slips once, at the earliest of its
    arrivals, over its rise time. The data, ordered by component, station and bin, are linear in
    the slips of the patches along the two rakes of the static green's functions, which integrate
    out exactly under their prior N(0, sigma^2 C0); see {altar.models.seismic.marginal}.

    The slip prior must be gaussian with zero mean, or that gaussian truncated to non-negative
    slips along one rake or both, {positive}, whose box probabilities are estimated by GHK; the
    magnitude and rake of polar slips, other bounds, or a moment prior have no closed form: sample
    the slips with kinematicmesh then. The noise: altar.data.datakron, or a constant cd_std,
    without a mask. On the cpu, or on cuda in double precision, job.gpuprecision = float64.

    My parameter sets, by name: risetime, one, one per patch, or one per group of {rise_groups};
    rupturevelocity, one per nucleation point; hypocenter, two per nucleation point, along strike
    and down dip in the surface coordinates of the mesh; origin, one per nucleation point, or none
    for one nucleation point at the time {origin}; and, with {slip_scale}, that set, one
    parameter, log10 sigma
    """


    # user configurable state
    mesh = altar.properties.path(default="mesh.h5")
    mesh.doc = "the patches: vertices, surface coordinates uv, faces, and the source points"

    Nmesh = altar.properties.int(default=10)
    Nmesh.doc = "the triangles of the mesh of the arrival times along each edge of a patch"

    green = altar.properties.path(default="static.gf.h5")
    green.doc = "the static green's functions, (2 patches, components x stations): the strike " \
                "slips of the patches, then their dip slips, 1 m each"

    components = altar.properties.int(default=3)
    components.doc = "the components of each station, e.g. east, north and up"

    bins = altar.properties.int(default=1)
    bins.doc = "the time bins of the data, of {dt} each, the first starting at time 0"

    dt = altar.properties.float(default=1.0)
    dt.doc = "the length of each bin, in the units of the origin times and the rise times"

    stf = altar.properties.str(default="triangle")
    stf.doc = "the slip rate of each source point over its rise time: triangle, or cosine"

    peak = altar.properties.float(default=0.5)
    peak.doc = "where the triangle peaks, as a fraction of the rise time"

    integration = altar.properties.str(default="bin")
    integration.doc = "the data of each bin: bin, the default, the average over it, for binned " \
                      "data; step, at its end"

    rise_groups = altar.properties.str(default=None)
    rise_groups.doc = "the dataset of {mesh} with the rise time group of each patch, for as many " \
                      "rise times as groups"

    origin = altar.properties.float(default=0.0)
    origin.doc = "the origin time of the one nucleation point, without an origin parameter set"

    slip_sigma = altar.properties.float(default=1.0)
    slip_sigma.doc = "sigma, the standard deviation of the prior of the slips, unless sampled"

    slip_scale = altar.properties.str(default=None)
    slip_scale.doc = "the parameter set of log10 sigma, if sampled"

    cm_file = altar.properties.path(default=None)
    cm_file.doc = "C0, (2 patches, 2 patches), the covariance of the slips over sigma^2; " \
                  "default the identity"

    positive = altar.properties.str(default=None)
    positive.doc = "the slips bounded below by 0, their prior truncated: strike, dip or both; " \
                   "default none"

    ghk_paths = altar.properties.int(default=256)
    ghk_paths.doc = "the paths of the GHK estimate of the probability of the bounded slips, " \
                    "the same sobol points for every sample"

    ghk_seed = altar.properties.int(default=0)
    ghk_seed.doc = "the seed of the scrambling of the sobol points of GHK"

    chunk = altar.properties.int(default=64)
    chunk.doc = "the samples whose likelihoods are computed at once, on the cpu"

    gpu_chunk = altar.properties.int(default=256)
    gpu_chunk.doc = "the samples whose likelihoods are computed at once on cuda, each with a " \
                    "(2 patches)^2 system"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Load the mesh, the green's functions, the data and their noise; lay out my parameters
        """
        cuda = altar.backends.active() == "cuda"
        if cuda and application.job.gpuprecision != "float64":
            application.error.log("kinematicgaussianslip: on cuda in double precision, job.gpuprecision = float64")
            raise SystemExit(1)
        super().initialize(application=application)
        if cuda and self.embedded:
            application.error.log("kinematicgaussianslip: not in an ensemble on cuda")
            raise SystemExit(1)
        import h5py
        from . import mesh as meshes
        from .marginal import GaussianSlip
        from altar.data.Kronecker import Kronecker
        error = application.error

        # the patches and their source points
        with h5py.File(self.ifs[str(self.mesh)].uri.path, "r") as h5:
            vertices = numpy.asarray(h5["vertices"], dtype=float)
            uv = numpy.asarray(h5["uv"], dtype=float)
            faces = numpy.asarray(h5["faces"], dtype=int)
            sources = numpy.asarray(h5["sources/uv"], dtype=float)
            owners = numpy.asarray(h5["sources/patch"], dtype=int)
            groups = None if self.rise_groups is None else numpy.asarray(h5[self.rise_groups], dtype=int)
        Np = faces.shape[0]
        counts = numpy.bincount(owners, minlength=Np)
        if counts.min() != counts.max() or numpy.any(numpy.diff(owners) < 0):
            error.log(f"'{self.mesh}': each patch needs the same number of source points, stored patch by patch")
            raise SystemExit(1)
        self.patches, self.points = Np, int(counts[0])

        # my parameters
        names = ["risetime", "rupturevelocity", "hypocenter"]
        names += [] if self.slip_scale is None else [self.slip_scale]
        missing = [name for name in names if name not in self.psets_list]
        if missing:
            error.log(f"kinematicgaussianslip: no parameter sets {missing}")
            raise SystemExit(1)
        column = lambda name: self.psets[name].offset + numpy.arange(self.psets[name].count)
        self._velocity = column("rupturevelocity")
        self._hypocenter, self._rise = column("hypocenter"), column("risetime")
        # without origins, one nucleation point at a known time
        self._origin = column("origin") if "origin" in self.psets_list else None
        self.fronts = self._velocity.size
        origins = self.fronts if self._origin is None else self._origin.size
        if (self._origin is None and self.fronts > 1) or origins != self.fronts \
                or self._hypocenter.size != 2 * self.fronts:
            error.log(f"kinematicgaussianslip: for {self.fronts} nucleation points, as many origins and "
                      f"rupture velocities and twice as many hypocenter coordinates; without origins, one")
            raise SystemExit(1)
        rises = self._rise.size
        if groups is None:
            groups = numpy.zeros(Np, dtype=int) if rises == 1 else numpy.arange(Np)
        if groups.size != Np or groups.min() < 0 or numpy.unique(groups).size != rises or groups.max() + 1 != rises:
            error.log(f"kinematicgaussianslip: {rises} rise times: one, one per patch, or one per "
                      f"group of '{self.rise_groups}', numbered from 0")
            raise SystemExit(1)
        self._groups = groups
        self._scale = None if self.slip_scale is None else column(self.slip_scale)
        if self._scale is not None and self._scale.size != 1:
            error.log(f"kinematicgaussianslip: '{self.slip_scale}' holds one parameter, log10 sigma")
            raise SystemExit(1)
        if self.integration not in ("bin", "step") or self.stf not in ("triangle", "cosine") \
                or not 0 < self.peak < 1:
            error.log("kinematicgaussianslip: integration bin or step; stf triangle or cosine, a peak in (0, 1)")
            raise SystemExit(1)

        # the noise, separable in space and time
        data = self.dataobs
        if data.mask is not None:
            error.log("kinematicgaussianslip: no mask; fill or drop the missing bins first")
            raise SystemExit(1)
        stations, remainder = divmod(self.observations, self.components * self.bins)
        if remainder:
            error.log(f"kinematicgaussianslip: {self.observations} observations aren't {self.components} "
                      f"components x stations x {self.bins} bins")
            raise SystemExit(1)
        cd_inv = data.cd_inv
        if isinstance(cd_inv, float):
            σ = 1 / cd_inv
            noise = Kronecker(numpy.array([σ * σ * numpy.eye(stations)] * self.components),
                              numpy.array([numpy.eye(self.bins)] * self.components))
        elif hasattr(cd_inv, "covariance"):
            noise = cd_inv.covariance
        else:
            error.log("kinematicgaussianslip: the noise of altar.data.datakron, or a constant cd_std")
            raise SystemExit(1)
        if (noise.components, noise.stations, noise.bins) != (self.components, stations, self.bins):
            error.log(f"kinematicgaussianslip: the noise is {noise.components} x {noise.stations} x "
                      f"{noise.bins}, not {self.components} x {stations} x {self.bins}")
            raise SystemExit(1)
        self.stations = stations

        # the prior of the slips, and the marginal likelihood
        G = self.io.load(filename=self.green, shape=(2 * Np, self.components * stations))
        C0 = None if self.cm_file is None else self.io.load(filename=self.cm_file, shape=(2 * Np, 2 * Np))
        sides = {None: [], "strike": [0], "dip": [1], "both": [0, 1]}
        if self.positive not in sides:
            error.log(f"kinematicgaussianslip: positive, strike, dip or both; got '{self.positive}'")
            raise SystemExit(1)
        mask = numpy.zeros((2, Np), bool)
        mask[sides[self.positive]] = True
        self.marginal = GaussianSlip(G, data.observed(), noise, C0=C0, positive=mask.ravel(),
                                     paths=self.ghk_paths, seed=self.ghk_seed)

        # the arrival times
        from altar.models.seismic.ext import seismic
        grid = meshes.arrival_mesh(vertices, faces, uv, sources, owners, self.Nmesh)
        self._corners, self._weights = grid["corners"], grid["weights"]
        self.cmodel = seismic.KinematicMesh(
            Np=Np, Nt=1, dt=1.0, t0s=numpy.zeros(Np), Nsp=self.points,
            idx_map=numpy.arange(4 * Np + 2, dtype="int64"), precision="float64", **grid)
        # the patches in surface coordinates, for the hypocenters on the mesh
        U = uv[faces]
        self._triangle_origin = U[:, 0]
        self._triangle_inverse = numpy.linalg.inv(numpy.stack([U[:, 1] - U[:, 0], U[:, 2] - U[:, 0]], axis=2))
        if cuda:
            self._cuda(grid)
        application.info.log(
            f"kinematic gaussian slip: {Np} patches, {self.points} source points each, {self.fronts} "
            f"nucleation points, {rises} rise times; {self.components} x {stations} x {self.bins} data")
        return self


    def verify_theta(self, theta, mask, batch=None):
        """
        Reject, besides the bounds of my parameter sets, the samples with their origins out of
        order, the nucleation points being interchangeable, or a hypocenter off the mesh
        """
        super().verify_theta(theta=theta, mask=mask, batch=batch)
        θ = numpy.asarray(theta if self._gpu is not None else self.restrict(theta=theta), dtype=float)
        n = θ.shape[0] if batch is None else batch
        θ = θ[:n]
        bad = numpy.zeros(n, dtype=bool)
        if self._origin is not None and self.fronts > 1:
            bad |= numpy.any(numpy.diff(θ[:, self._origin], axis=1) < 0, axis=1)
        for k in range(self.fronts):
            bad |= ~self.on_mesh(θ[:, self._hypocenter[2 * k:2 * k + 2]])
        numpy.asarray(mask)[:n] += bad
        return mask


    def eval_data_likelihood(self, theta, likelihood, batch=None):
        """
        The marginal log likelihood of the data of each sample, the slips integrated out
        """
        if self._gpu is not None:
            n = theta.shape[0] if batch is None else batch
            for k, mesh in enumerate(self._meshes):
                mesh.source_arrivals(theta=theta.grid, out=self._TI.grid, batch=n, offset=k * self._stride)
            self._gpu.loglikelihood(theta=theta.grid, TI=self._TI.grid, stride=self._stride,
                                    out=likelihood.grid, batch=n)
            return self
        θ = numpy.asarray(self.restrict(theta=theta), dtype=float)
        n = θ.shape[0] if batch is None else batch
        out = numpy.empty(n)
        for start in range(0, n, self.chunk):
            rows = θ[start:min(n, start + self.chunk)]
            out[start:start + rows.shape[0]] = self.marginal.loglikelihood(self.fractions(rows), self.sigma(rows))
        numpy.asarray(likelihood)[:n] = out
        return self


    @altar.export
    def forward_problem(self, application, theta):
        """
        For each row of {theta}: the data of the mean of the slips given the data, "data"; that
        mean, "slip", the strike slips of the patches then their dip slips; and the cumulative
        slip fraction of each patch in each bin, "F"
        """
        θ = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        F = self.fractions(θ)
        mean = self.marginal.mean(F, self.sigma(θ), numpy.random.default_rng(self.ghk_seed))
        return {"data": self.marginal.predict(F, mean), "slip": mean, "F": F}


    def draw_slips(self, theta, rng, count=1):
        """
        {count} draws of the slips given the data for each row of {theta}, (rows, count, 2 patches)
        """
        θ = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        return self.marginal.draw(self.fractions(θ), self.sigma(θ), rng, count)


    # implementation details
    def _cuda(self, grid):
        """
        The marginal likelihood on the device: the arrival times of each nucleation point, by a
        cuda kinematic model reading its velocity and hypocenter from my columns, then the rest
        """
        import math
        import altar.cuda
        from altar.models.seismic.ext import cudaseismic
        Np, m = self.patches, self.marginal
        self._t0s = altar.cuda.vector(source=numpy.zeros(Np), dtype="float64")
        self._meshes, self._maps = [], []
        for k in range(self.fronts):
            idx = numpy.zeros(4 * Np + 2, dtype="int64")
            idx[3 * Np:4 * Np] = self._velocity[k]
            idx[4 * Np:] = self._hypocenter[2 * k:2 * k + 2]
            self._maps.append(altar.cuda.vector(source=idx))
            self._meshes.append(cudaseismic.cudaKinematicMesh(
                Np=Np, Nt=1, dt=1.0, t0s=self._t0s.grid, Nsp=self.points, samples=self.samples,
                parameters=self.parameters, observations=1, idx_map=self._maps[-1].grid, **grid))
        self._stride = self.samples * Np * self.points
        self._TI = altar.cuda.matrix(shape=(self.fronts * self.samples, Np * self.points), dtype="float64")
        constant = -0.5 * (m.quadratic + m.logdetC0 + m.noise.logdet + m.observations * math.log(2 * math.pi)) - m.logZ0
        uniforms = m.uniforms if m.bounded else numpy.zeros(0)
        self._gpu = cudaseismic.cudaGaussianSlip(
            Np=Np, Nsp=self.points, bins=self.bins, components=self.components, dt=self.dt,
            kind={"triangle": 0, "cosine": 1}[self.stf], peak=self.peak,
            mode={"step": 1, "bin": 2}[self.integration],
            origin_columns=numpy.array([-1] if self._origin is None else self._origin, dtype="int64"),
            origin=self.origin, rise_columns=self._rise[self._groups].astype("int64"),
            scale_column=-1 if self._scale is None else int(self._scale[0]), sigma=self.slip_sigma,
            GG=numpy.array(m.GG), GSD=numpy.array(m.GSD), Rinv=numpy.array(m.Rinv), C0inv=m.C0inv,
            order=m.order.astype("int32"), bounded=m.bounded, uniforms=uniforms, constant=constant,
            chunk=min(self.gpu_chunk, self.samples))
        return self


    def sigma(self, θ):
        """
        The standard deviation of the prior of the slips of each row of {θ}
        """
        if self._scale is None:
            return numpy.full(θ.shape[0], float(self.slip_sigma))
        return 10.0 ** θ[:, self._scale[0]]


    def fractions(self, θ):
        """
        The cumulative slip fraction of each patch in each bin, (rows, patches, bins)
        """
        from .marginal import fractions
        onsets = self.onsets(θ)
        risetime = θ[:, self._rise][:, self._groups]
        return fractions(onsets, risetime, self.dt, self.bins, self.stf, self.peak, self.integration)


    def onsets(self, θ):
        """
        The earliest arrival of the fronts at each source point, (rows, patches, points)
        """
        n, Np = θ.shape[0], self.patches
        onsets = None
        for k in range(self.fronts):
            rows = numpy.zeros((n, 4 * Np + 2))
            rows[:, 2 * Np:3 * Np] = 1
            rows[:, 3 * Np:4 * Np] = θ[:, self._velocity[k]][:, None]
            rows[:, 4 * Np:] = θ[:, self._hypocenter[2 * k:2 * k + 2]]
            T = numpy.asarray(self.cmodel.arrivals(theta=numpy.ascontiguousarray(rows), batch=n)).reshape(n, -1)
            origin = self.origin if self._origin is None else θ[:, self._origin[k]][:, None]
            times = (self._weights * T[:, self._corners]).sum(-1) + origin
            onsets = times if onsets is None else numpy.minimum(onsets, times)
        return onsets.reshape(n, Np, self.points)


    def on_mesh(self, uv):
        """
        Whether each point {uv}, (rows, 2), is on a patch, in surface coordinates
        """
        b = numpy.einsum("tij,stj->sti", self._triangle_inverse, uv[:, None, :] - self._triangle_origin[None])
        bary = numpy.concatenate([1 - b.sum(axis=2, keepdims=True), b], axis=2)
        return bary.min(axis=2).max(axis=1) >= -1e-9


    # private data
    marginal = None # the likelihood of the fronts, the slips integrated out
    _gpu = None # its twin on cuda
    cmodel = None # the compiled model, for the arrival times
    fronts = None # the nucleation points
    patches = None
    points = None # the source points of each patch
    stations = None


# end of file
