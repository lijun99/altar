# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
from __future__ import annotations
import math
import os
import typing
import numpy

if typing.TYPE_CHECKING:
    from altar.shells.Application import Application
    from altar.simulations.NumpyRNG import NumpyRNG

from .Base import Base as base


# the correlation of two parameters a distance r apart, r in units of the correlation length
KERNELS = {
    "exponential": lambda r: numpy.exp(-r),
    "matern32": lambda r: (1 + math.sqrt(3) * r) * numpy.exp(-math.sqrt(3) * r),
    "gaussian": lambda r: numpy.exp(-0.5 * r * r),
}


class Correlated(base):
    """
    The cpu implementation of a gaussian distribution whose parameters are correlated in
    space: N(mean, C), with C_ij = sigma^2 k(|x_i - x_j| / length) for the locations x_i of
    the parameters, e.g. the centroids of the patches of a fault, and a correlation kernel k;
    optionally restricted to a support, e.g. to keep the slip positive, and normalized within it
    by the GHK estimate of its mass
    """


    def initialize(self, rng: NumpyRNG, application: Application | None = None) -> typing.Self:
        """
        Build the covariance from the locations of my parameters, and factor it
        """
        # hold on to the generator
        super().initialize(rng=rng, application=application)
        # the covariance, its factor and its inverse
        self.build(application=application)
        # set up my transform, if reparameterizing
        self._initialize_transform(application=application)
        # all done
        return self


    def build(self, application: Application | None = None) -> typing.Self:
        """
        The covariance of my parameters, C = L L^T, L^-1 and C^-1, from their locations
        """
        if self.kernel not in KERNELS:
            raise ValueError(f"correlated: unknown kernel '{self.kernel}'; "
                             f"one of {', '.join(sorted(KERNELS))}")
        x = load_coordinates(distribution=self, application=application)
        if x.shape[0] != self.parameters:
            raise ValueError(f"correlated: {x.shape[0]} locations in '{self.coordinates}' "
                             f"for {self.parameters} parameters")
        length = numpy.broadcast_to(numpy.asarray(self.length, dtype=float).ravel(), (x.shape[1],))
        if numpy.any(length <= 0):
            raise ValueError(f"correlated: the correlation length must be positive, got {self.length}")
        # the distances between the parameters, in units of the correlation lengths
        d = (x[:, None, :] - x[None, :, :]) / length
        r = numpy.sqrt((d * d).sum(axis=-1))
        n = self.parameters
        self.covariance = self.sigma ** 2 * (KERNELS[self.kernel](r) + self.jitter * numpy.identity(n))
        self.factor = numpy.linalg.cholesky(self.covariance)
        self.whitener = numpy.linalg.inv(self.factor)
        self.precision_matrix = self.whitener.T @ self.whitener
        self.mean_vector = numpy.full(n, float(self.mean))
        # the normalization of the untruncated distribution, -n/2 log 2π - log |L|
        self.constant = -0.5 * n * math.log(2 * math.pi) - numpy.log(numpy.diag(self.factor)).sum()
        low, high = self.support
        self.bounded = bool(numpy.isfinite(low) or numpy.isfinite(high))
        # within a support, less the log of its mass, so the density integrates to 1 over it
        if self.bounded and self.normalization_draws > 0:
            self.log_mass, self.log_mass_error = self.mass(draws=self.normalization_draws)
            self.constant -= self.log_mass
            info = getattr(application, "info", None)
            if info is not None:
                info.log(f"correlated prior: the support holds exp({self.log_mass:.6g} ± "
                         f"{self.log_mass_error:.2g}) of the untruncated distribution")
        return self


    def mass(self, draws: int) -> tuple[float, float]:
        """
        log P(low <= X <= high) for X ~ N(mean, C), and its standard error, by the GHK simulator
        (Geweke, Hajivassiliou and Keane): each coordinate drawn in turn from its truncated
        conditional along the Cholesky factor, the product of their masses averaged; with a
        fixed seed, so the normalization is the same in every run
        """
        low, high = self.support
        L, m, n = self.factor, self.mean_vector, self.parameters
        rng = numpy.random.default_rng(1)
        z = numpy.zeros((draws, n))
        logw = numpy.zeros(draws)
        for i in range(n):
            shift = m[i] + z[:, :i] @ L[i, :i]
            a, b = (low - shift) / L[i, i], (high - shift) / L[i, i]
            # in the upper tail, by symmetry, so the masses don't cancel
            upper = a > 0
            lo = numpy.where(upper, _ndtr(-b), _ndtr(a))
            p = numpy.where(upper, _ndtr(-a), _ndtr(b)) - lo
            logw += numpy.log(numpy.maximum(p, 1e-300))
            x = _ndtri(lo + rng.uniform(size=draws) * p)
            z[:, i] = numpy.where(upper, -x, x)
        top = logw.max()
        w = numpy.exp(logw - top)
        return float(top + numpy.log(w.mean())), float(w.std() / (w.mean() * math.sqrt(draws)))


    def verify(self, theta: numpy.ndarray, mask: numpy.ndarray,
               batch: int | None = None) -> numpy.ndarray:
        """
        Mark in {mask} the samples with a parameter outside my support; {theta} is physical,
        reparameterized or not
        """
        if not self.bounded:
            return mask
        return self.outside(theta=theta, mask=mask, support=self.support)


    def draw(self, shape: tuple[int, ...]) -> numpy.ndarray:
        """
        Samples of my distribution, one per row of {shape}: exact draws without bounds, and by
        elliptical slice sampling (Murray et al., 2010) within them
        """
        rows = shape[0] if len(shape) == 2 else 1
        x = self.mean_vector + self.rng.standard_normal((rows, self.parameters)) @ self.factor.T
        if self.bounded:
            x = self.truncated(start=x)
        return x if len(shape) == 2 else x[0]


    def truncated(self, start: numpy.ndarray) -> numpy.ndarray:
        """
        Samples of my distribution within my support, from {sweeps} sweeps of elliptical slice
        sampling from {start}, moved inside the support first
        """
        low, high = self.support
        rows, n = start.shape
        m = self.mean_vector
        # a start strictly inside the support
        width = high - low if numpy.isfinite(high - low) else 1.0
        x = numpy.clip(start, low + 1e-9 * width, high - 1e-9 * width)
        inside = lambda y: ((y >= low) & (y <= high)).all(axis=1)
        for _ in range(self.sweeps):
            nu = self.rng.standard_normal((rows, n)) @ self.factor.T
            angle = self.rng.uniform(0, 2 * math.pi, size=rows)
            lo, hi = angle - 2 * math.pi, angle.copy()
            d = x - m
            pending = numpy.arange(rows)
            # shrink the bracket of each row until its proposal is inside; it closes on the
            # current sample, which is, so a row still pending after this keeps it
            for _ in range(64):
                if pending.size == 0:
                    break
                a = angle[pending]
                proposal = m + d[pending] * numpy.cos(a)[:, None] + nu[pending] * numpy.sin(a)[:, None]
                ok = inside(proposal)
                x[pending[ok]] = proposal[ok]
                bad = pending[~ok]
                below = angle[bad] < 0
                lo[bad[below]] = angle[bad[below]]
                hi[bad[~below]] = angle[bad[~below]]
                angle[bad] = self.rng.uniform(lo[bad], hi[bad])
                pending = bad
        return x


    def log_density(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        The log density of each row of {x}, -inf outside my support
        """
        x = numpy.atleast_2d(numpy.asarray(x, dtype=numpy.float64))
        z = (x - self.mean_vector) @ self.whitener.T
        density = self.constant - 0.5 * (z * z).sum(axis=1)
        if self.bounded:
            low, high = self.support
            density = numpy.where(((x >= low) & (x <= high)).all(axis=1), density, -numpy.inf)
        return density


    def eval_prior(self, theta: numpy.ndarray, likelihood: numpy.ndarray,
                   batch: int | None = None) -> typing.Self:
        """
        Add to {likelihood} the log prior probabilities of my portion of the samples in {theta}
        """
        θ = self.restrict(theta=theta)
        likelihood[:θ.shape[0]] += self.log_density(θ)
        return self


    def prior_gradient(self, theta: numpy.ndarray, gradient: numpy.ndarray,
                       batch: int | None = None) -> typing.Self:
        r"""
        Fill my portion of {gradient} with d\log P/d\theta = -C^-1 (\theta - mean); when
        reparameterized, its chain rule to the sampling space, with the gradient of log|J|
        """
        θ = numpy.asarray(self.restrict(theta=theta), dtype=numpy.float64)
        g = self.restrict(theta=gradient)
        g[...] = (self.mean_vector - θ) @ self.precision_matrix
        if self.reparameterize:
            self.transform.chain_gradient(theta=self.restrict(theta=theta), gradient=g, batch=batch)
        return self


    # private data, set by the shim before {initialize} runs
    mean: float
    sigma: float
    kernel: str
    length: typing.Any
    coordinates: str
    dataset: str | None
    jitter: float
    sweeps: int
    normalization_draws: int = 10000
    # set by {build}
    bounded: bool = False
    log_mass: float = 0.0 # log P(X in my support) of the untruncated distribution
    log_mass_error: float = 0.0
    covariance: numpy.ndarray
    factor: numpy.ndarray # L, with C = L L^T
    whitener: numpy.ndarray # L^-1
    precision_matrix: numpy.ndarray # C^-1
    mean_vector: numpy.ndarray
    constant: float


_erfc = numpy.frompyfunc(math.erfc, 1, 1)


def _ndtr(x: numpy.ndarray) -> numpy.ndarray:
    """
    The standard normal cumulative distribution function, accurate in the lower tail
    """
    return _erfc(-numpy.asarray(x, dtype=float) / math.sqrt(2)).astype(float) / 2


def _ndtri(p: numpy.ndarray) -> numpy.ndarray:
    """
    The inverse of {_ndtr}: Acklam's rational approximation, refined by one Halley step
    """
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    poly = lambda coefficients, t: numpy.polyval(coefficients, t)
    p = numpy.clip(numpy.asarray(p, dtype=float), 1e-300, 1 - 1e-16)
    q = p - 0.5
    r = q * q
    x = q * poly(a, r) / poly(b + (1.0,), r)
    tail = numpy.sqrt(-2 * numpy.log(numpy.minimum(p, 1 - p)))
    edge = poly(c, tail) / poly(d + (1.0,), tail)
    x = numpy.where(p < 0.02425, edge, numpy.where(p > 1 - 0.02425, -edge, x))
    e = _ndtr(x) - p
    u = e * math.sqrt(2 * math.pi) * numpy.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def load_coordinates(distribution: typing.Any, application: typing.Any = None) -> numpy.ndarray:
    """
    The locations of the parameters of {distribution}, (parameters x dimensions), from its
    {coordinates} file: .h5, its {dataset} or the first one, or text; relative to the current
    directory, or to the model's input directory
    """
    if distribution.coordinates is None:
        raise ValueError("correlated: 'coordinates', the locations of the parameters, is required")
    path = str(distribution.coordinates)
    if not os.path.exists(path) and application is not None:
        path = application.pfs["inputs"][path].uri.path
    if path.endswith(".h5"):
        import h5py
        with h5py.File(path, "r") as h5:
            name = distribution.dataset or next(iter(h5))
            x = numpy.asarray(h5[name], dtype=float)
    else:
        x = numpy.loadtxt(path, dtype=float, ndmin=2)
    return x.reshape(x.shape[0], -1)


# end of file
