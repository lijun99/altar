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
from altar.models.BayesianL2 import BayesianL2


# declaration
class Arrivals(BayesianL2, family="altar.models.seismic.arrivals"):
    """
    The arrival times of a phase, e.g. P, at stations, from the hypocenter on a triangulated fault:
    the hypocenter in the surface coordinates of the mesh, as the kinematic model on triangles
    has it, mapped to space through the triangle holding it; the travel times through a
    homogeneous medium, or from a table of travel times against depth and distance. In an
    ensemble with the kinematic model, the two share the hypocenter, which the arrival times pin
    down apart from the rupture velocities.

    My parameters are the hypocenter along strike and down dip; the picks, my data, are in the
    time of the kinematic model, its origin time 0, plus {origin}.
    """


    # user configurable state
    mesh = altar.properties.path(default="mesh.h5")
    mesh.doc = "the patches: vertices (km, depth positive), surface coordinates uv and faces"

    stations = altar.properties.path(default="stations.txt")
    stations.doc = "the stations, one per pick: x and y (km), in the frame of the mesh, and " \
                   "optionally their elevation (km); a text or an .h5 file"

    velocity = altar.properties.float(default=6.0)
    velocity.doc = "the velocity of the phase (km/s), for straight rays in a homogeneous medium"

    table = altar.properties.path(default=None)
    table.doc = "an .h5 file with the travel times of the phase, 'times' (s), on the grid of " \
                "'depths' and 'distances' (km); default straight rays at {velocity}"

    origin = altar.properties.float(default=0.0)
    origin.doc = "the time of the origin of the rupture, in the time of the picks (s)"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Load the mesh, the stations and the travel times
        """
        super().initialize(application=application)
        self.return_residual = False
        import h5py
        with h5py.File(self.ifs[str(self.mesh)].uri.path, "r") as h5:
            xyz = numpy.asarray(h5["vertices"], dtype=float)
            uv = numpy.asarray(h5["uv"], dtype=float)
            faces = numpy.asarray(h5["faces"], dtype=int)
        # the affine map of each triangle from its surface coordinates to space, x = M u + c
        X, U = xyz[faces], uv[faces]
        A = numpy.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]], axis=2)
        B = numpy.stack([U[:, 1] - U[:, 0], U[:, 2] - U[:, 0]], axis=2)
        self._map = A @ numpy.linalg.inv(B)
        self._shift = X[:, 0] - numpy.einsum("tij,tj->ti", self._map, U[:, 0])
        self._corners, self._inverse = U[:, 0], numpy.linalg.inv(B)
        # the stations, at the surface unless given an elevation
        path = self.ifs[str(self.stations)].uri.path
        if str(self.stations).endswith(".h5"):
            with h5py.File(path, "r") as h5:
                s = numpy.asarray(h5[next(iter(h5))], dtype=float)
        else:
            s = numpy.loadtxt(path, dtype=float, ndmin=2)
        if s.shape[0] != self.observations:
            self.error.log(f"'{self.stations}': {s.shape[0]} stations for {self.observations} picks")
            raise SystemExit(1)
        self._stations = numpy.column_stack([s[:, :2], -s[:, 2] if s.shape[1] > 2 else numpy.zeros(len(s))])
        # the travel times
        self._table = None
        if self.table is not None:
            with h5py.File(self.ifs[str(self.table)].uri.path, "r") as h5:
                self._table = tuple(numpy.asarray(h5[k], dtype=float) for k in ["depths", "distances", "times"])
        return self


    def forward_model_batched(self, theta, prediction, batch=None):
        """
        Fill the first {batch} rows of {prediction}, (samples x picks), with the arrival times of
        the hypocenters in {theta}
        """
        θ = numpy.array(theta, dtype=float)
        batch = θ.shape[0] if batch is None else batch
        times, _ = self.arrivals(θ[:batch])
        numpy.asarray(prediction)[:batch] = times
        return self


    @altar.export
    def forward_problem(self, application, theta):
        """
        The arrival times for each row of {theta}; see {altar.models.Model}
        """
        times, _ = self.arrivals(numpy.atleast_2d(numpy.asarray(theta, dtype=float)))
        return {"data": times}


    def gradient(self, controller, step, batch=None):
        """
        Fill {step.data_gradient} with the gradient of my data log likelihood, (d - t)^T C^-1 dt/dθ
        """
        # in an ensemble, the ensemble owns the parameter sets and their priors
        if not self.embedded:
            if not self.checked_unbounded_priors:
                self.verify_unbounded_priors()
            for name in self.psets_list:
                self.psets[name].prior_gradient(theta=step.theta, gradient=step.prior_gradient, batch=batch)
        θ = numpy.array(step.theta, dtype=float)
        rows = θ.shape[0]
        batch = rows if batch is None else batch
        times, jacobian = self.arrivals(θ[:batch])
        residual = self.dataobs.observed()[None, :] - times
        covariance = self.dataobs.covariance()
        w = residual / covariance if numpy.ndim(covariance) == 0 else numpy.linalg.solve(covariance, residual.T).T
        gradient = numpy.zeros_like(θ)
        gradient[:batch] = numpy.einsum("so,sop->sp", w, jacobian)
        numpy.asarray(step.data_gradient)[...] = gradient
        return self


    # implementation details
    def arrivals(self, theta):
        """
        The arrival times, (samples x picks), of the hypocenters in {theta}, (samples x 2), and
        their derivatives with respect to the hypocenter, (samples x picks x 2)
        """
        uv = theta[:, :2]
        # the triangle of each hypocenter: the one it is inside of, or the nearest
        b = numpy.einsum("tij,stj->sti", self._inverse, uv[:, None, :] - self._corners[None])
        bary = numpy.concatenate([1 - b.sum(axis=2, keepdims=True), b], axis=2)
        triangle = numpy.argmax(bary.min(axis=2), axis=1)
        M = self._map[triangle]
        x = numpy.einsum("sij,sj->si", M, uv) + self._shift[triangle]
        # the straight path to each station, or the depth and the horizontal distance
        d = x[:, None, :] - self._stations[None]
        if self._table is None:
            r = numpy.sqrt((d * d).sum(axis=2))
            times = r / self.velocity
            dtdx = d / (r[..., None] * self.velocity)
        else:
            depth = x[:, 2][:, None] - self._stations[None, :, 2]
            horizontal = numpy.sqrt((d[..., :2] ** 2).sum(axis=2))
            times, dtdz, dtdr = _bilinear(*self._table, depth, horizontal)
            dtdx = numpy.concatenate([dtdr[..., None] * d[..., :2] / numpy.maximum(horizontal, 1e-9)[..., None],
                                      dtdz[..., None]], axis=2)
        return times + self.origin, numpy.einsum("sok,skj->soj", dtdx, M)


def _bilinear(depths, distances, times, z, r):
    """
    The bilinear interpolation of {times} on the grid of {depths} and {distances} at {z} and {r},
    and its derivatives with respect to them
    """
    i = numpy.clip(numpy.searchsorted(depths, z) - 1, 0, len(depths) - 2)
    j = numpy.clip(numpy.searchsorted(distances, r) - 1, 0, len(distances) - 2)
    dz, dr = depths[i + 1] - depths[i], distances[j + 1] - distances[j]
    a, c = (z - depths[i]) / dz, (r - distances[j]) / dr
    t00, t01, t10, t11 = times[i, j], times[i, j + 1], times[i + 1, j], times[i + 1, j + 1]
    t = (1 - a) * (1 - c) * t00 + (1 - a) * c * t01 + a * (1 - c) * t10 + a * c * t11
    dtdz = ((1 - c) * (t10 - t00) + c * (t11 - t01)) / dz
    dtdr = ((1 - a) * (t01 - t00) + a * (t11 - t10)) / dr
    return t, dtdz, dtdr


# end of file
