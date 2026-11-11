# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


from __future__ import annotations
import math
import typing
import numpy
from altar.bayesian.schedulers.ConstantTemperature import ConstantTemperature
from altar.bayesian.stepcounters.StepCounter import walk
from altar.distributions.native.Correlated import box_mass

if typing.TYPE_CHECKING:
    import journal
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.states.CoolingStep import CoolingStep
    from altar.bayesian.stepcounters.StepCounter import StepCounter
    from altar.shells.Application import Application


class Statistics(typing.NamedTuple):
    accepted: int
    invalid: int
    rejected: int


class Posterior:
    """
    The posterior of a linear model with gaussian noise and a gaussian prior on a box, at β:
    N(mean, A^-1) on [low, high], with A = P + β Gw^T Gw and A mean = P μ + β Gw^T dw, for the
    whitened system (Gw, dw) and the prior N(μ, P^-1), P zero where the prior is flat
    """

    def __init__(self, Gw: numpy.ndarray, dw: numpy.ndarray, mu: numpy.ndarray,
                 precision: numpy.ndarray, low: numpy.ndarray, high: numpy.ndarray,
                 beta: float) -> None:
        n = mu.size
        self.beta, self.low, self.high = beta, low, high
        A = precision + beta * (Gw.T @ Gw)
        try:
            L = numpy.linalg.cholesky(A)
        except numpy.linalg.LinAlgError:
            raise ValueError("the posterior precision is singular: some parameters have neither a "
                             "gaussian prior nor data that constrain them") from None
        # A = L L^T, A^-1 = L^-T L^-1
        self.factor = L
        self.whitener = numpy.linalg.solve(L, numpy.identity(n))
        self.covariance = self.whitener.T @ self.whitener
        self.mean = self.covariance @ (precision @ mu + beta * (Gw.T @ dw))
        # the perturbations of the prior, N(0, P), on the parameters it constrains
        self.constrained = numpy.flatnonzero(numpy.diag(precision) > 0)
        block = precision[numpy.ix_(self.constrained, self.constrained)]
        self.prior_factor = numpy.linalg.cholesky(block) if self.constrained.size else None
        # and of the data, N(0, Gw^T Gw), through R from Gw = Q R when it is the smaller
        self.data_factor = numpy.linalg.qr(Gw, mode="r").T if Gw.shape[0] > n else Gw.T
        # the coordinates with a finite bound
        self.bounded = bool(numpy.isfinite(low).any() or numpy.isfinite(high).any())
        return


    def perturbation(self, rows: int, rng: numpy.random.Generator) -> numpy.ndarray:
        """
        {rows} draws of N(0, A^-1) by perturbation: A x = ζ + √β ε, with ζ ~ N(0, P) the
        perturbation of the prior and ε ~ N(0, Gw^T Gw) that of the data
        """
        n = self.mean.size
        rhs = math.sqrt(self.beta) * rng.standard_normal((rows, self.data_factor.shape[1])) @ self.data_factor.T
        if self.prior_factor is not None:
            rhs[:, self.constrained] += (rng.standard_normal((rows, self.constrained.size))
                                         @ self.prior_factor.T)
        # the solve with the factor of A
        return (rhs @ self.whitener.T) @ self.whitener


    def trajectory(self, theta: numpy.ndarray, time: float, rng: numpy.random.Generator,
                   max_bounces: int) -> int:
        """
        Move each row of {theta} along an exact hamiltonian trajectory of length {time} on the
        gaussian truncated to the box, reflecting off its walls (Pakman & Paninski, 2014), from a
        velocity drawn by perturbation; return the bounces
        """
        m, S = self.mean, self.covariance
        s = numpy.diag(S)
        # the walls, relative to the mean
        low, high = self.low - m, self.high - m
        # x - m = a cos t + v sin t
        a = theta - m
        v = self.perturbation(rows=theta.shape[0], rng=rng)
        left = numpy.full(theta.shape[0], time)
        active = numpy.arange(theta.shape[0])
        bounces = 0
        for _ in range(max_bounces):
            if active.size == 0:
                break
            aa, vv = a[active], v[active]
            r = numpy.hypot(aa, vv)
            phi = numpy.arctan2(vv, aa)
            with numpy.errstate(invalid="ignore", divide="ignore"):
                # r cos(t - φ) falls through a lower wall at t = φ + acos(low/r), rises through
                # an upper one at t = φ - acos(high/r)
                tl = numpy.mod(phi + numpy.arccos(low / r), 2 * math.pi)
                tu = numpy.mod(phi - numpy.arccos(high / r), 2 * math.pi)
            tl[~(numpy.abs(low) < r)] = numpy.inf
            tu[~(numpy.abs(high) < r)] = numpy.inf
            # grazing hits, at the wall a chain just left
            tl[tl < 1e-12] = numpy.inf
            tu[tu < 1e-12] = numpy.inf
            t = numpy.minimum(tl, tu)
            k = numpy.argmin(t, axis=1)
            rows = numpy.arange(active.size)
            tau = t[rows, k]
            hit = tau < left[active]
            tau = numpy.where(hit, tau, left[active])
            c, sn = numpy.cos(tau)[:, None], numpy.sin(tau)[:, None]
            aa, vv = aa * c + vv * sn, vv * c - aa * sn
            # reflect the velocities off the walls hit, in the metric of A, and pin the
            # positions on them
            h, kh = rows[hit], k[hit]
            vk = vv[h, kh]
            vv[h] -= (2 * vk / s[kh])[:, None] * S[kh]
            aa[h, kh] = numpy.where(tu[h, kh] < tl[h, kh], high[kh], low[kh])
            a[active], v[active] = aa, vv
            left[active] -= tau
            active = active[hit]
            bounces += h.size
        else:
            raise RuntimeError(f"the trajectories took more than {max_bounces} bounces")
        theta[...] = numpy.clip(m + a, self.low, self.high)
        return bounces


    def log_density(self, theta: numpy.ndarray) -> numpy.ndarray:
        """
        log N(theta; mean, A^-1), untruncated
        """
        z = (theta - self.mean) @ self.factor
        n = self.mean.size
        return (-0.5 * (z * z).sum(axis=1) - 0.5 * n * math.log(2 * math.pi)
                + numpy.log(numpy.diag(self.factor)).sum())


    def log_mass(self, draws: int) -> tuple[float, float]:
        """
        log P(low <= X <= high) for X ~ N(mean, A^-1), and its standard error, by GHK on the
        bounded coordinates
        """
        if not self.bounded:
            return 0.0, 0.0
        walls = numpy.flatnonzero(numpy.isfinite(self.low) | numpy.isfinite(self.high))
        S = self.covariance[numpy.ix_(walls, walls)]
        return box_mass(mean=self.mean[walls], factor=numpy.linalg.cholesky(S),
                        low=self.low[walls], high=self.high[walls], draws=draws)


def pcg(apply: typing.Callable, precondition: typing.Callable | None, rhs: numpy.ndarray,
        tolerance: float, max_iterations: int) -> tuple[numpy.ndarray, int, float]:
    """
    Solve A x = b for every row b of {rhs}, (k x n), by preconditioned conjugate gradients, all rows
    at once: {apply} and {precondition} map (k x n) to (k x n). Stop when the largest
    preconditioned residual sqrt(r^T z / r0^T z0), which doesn't depend on the coordinates, is
    below {tolerance}; return the solutions, the iterations and that residual
    """
    precondition = precondition or (lambda r: r)
    x = numpy.zeros_like(rhs)
    r = numpy.array(rhs, dtype=float, copy=True)
    z = precondition(r)
    p = z.copy()
    rz = numpy.einsum("ij,ij->i", r, z)
    rz0 = numpy.where(rz > 0, rz, 1.0)
    residual = float(numpy.sqrt(numpy.abs(rz / rz0)).max()) if rz.size else 0.0
    iterations = 0
    while residual >= tolerance and iterations < max_iterations:
        Ap = apply(p)
        alpha = rz / numpy.einsum("ij,ij->i", p, Ap)
        x += alpha[:, None] * p
        r -= alpha[:, None] * Ap
        z = precondition(r)
        rz_new = numpy.einsum("ij,ij->i", r, z)
        residual = float(numpy.sqrt(numpy.abs(rz_new / rz0)).max())
        p = z + (rz_new / numpy.where(rz != 0, rz, 1.0))[:, None] * p
        rz = rz_new
        iterations += 1
    return x, iterations, residual


class DenseOperator:
    """
    The {LinearOperator} of a model that provides its dense whitened linear system and its gaussian
    prior: the matrix-free path on a problem small enough to also sample densely, to check one
    against the other. Unbounded only
    """

    def __init__(self, Gw: numpy.ndarray, dw: numpy.ndarray, mu: numpy.ndarray,
                 precision: numpy.ndarray, beta: float) -> None:
        self.Gw, self.beta, self.precision = Gw, beta, precision
        self.size = mu.size
        self.b = precision @ mu + beta * (Gw.T @ dw)
        self.constrained = numpy.flatnonzero(numpy.diag(precision) > 0)
        block = precision[numpy.ix_(self.constrained, self.constrained)]
        self.prior_factor = numpy.linalg.cholesky(block) if self.constrained.size else None
        # jacobi
        self.diagonal = numpy.diag(precision) + beta * (Gw * Gw).sum(axis=0)
        return

    def apply(self, x: numpy.ndarray) -> numpy.ndarray:
        return x @ self.precision + self.beta * ((x @ self.Gw.T) @ self.Gw)

    def rhs(self) -> numpy.ndarray:
        return self.b

    def perturb(self, rows: int, rng: numpy.random.Generator) -> numpy.ndarray:
        out = math.sqrt(self.beta) * rng.standard_normal((rows, self.Gw.shape[0])) @ self.Gw
        if self.prior_factor is not None:
            out[:, self.constrained] += (rng.standard_normal((rows, self.constrained.size))
                                         @ self.prior_factor.T)
        return out

    def precondition(self, r: numpy.ndarray) -> numpy.ndarray:
        return r / self.diagonal


class OperatorPosterior:
    """
    The posterior N(A^-1 b, A^-1) of a linear-gaussian model given as a {LinearOperator}, for
    problems too large to factor A: the mean and the perturbation draws are solves with A, by the
    operator's own {solve} if it has one, else by {pcg} with its {apply} and {precondition}. No
    bounds, and no evidence
    """

    bounded = False

    def __init__(self, operator: typing.Any, tolerance: float, max_iterations: int,
                 info: typing.Any) -> None:
        self.operator, self.tolerance, self.max_iterations, self.info = (
            operator, tolerance, max_iterations, info)
        if getattr(operator, "bounded", False):
            raise ValueError("the matrix-free path doesn't sample priors with bounds")
        self.mean = self.solve(numpy.atleast_2d(operator.rhs()), "mean")[0]
        return

    def solve(self, rhs: numpy.ndarray, what: str) -> numpy.ndarray:
        op = self.operator
        if hasattr(op, "solve"):
            x, iterations, residual = op.solve(rhs, self.tolerance, self.max_iterations)
        else:
            x, iterations, residual = pcg(op.apply, getattr(op, "precondition", None), rhs,
                                          self.tolerance, self.max_iterations)
        self.info.log(f"perturbation: {what}, {rhs.shape[0]} solves, {iterations} iterations, "
                      f"residual {residual:.2g}")
        if residual >= self.tolerance:
            raise ValueError(f"the solves didn't converge in {self.max_iterations} iterations "
                             f"(residual {residual:.2g})")
        return x

    def perturbation(self, rows: int, rng: numpy.random.Generator) -> numpy.ndarray:
        """
        {rows} draws of N(0, A^-1): A x = the operator's perturbation of the prior and the data
        """
        return self.solve(self.operator.perturb(rows, rng), "draws")

    log_density = None


class Perturbation:
    """
    The cpu implementation of the perturbation sampler of linear-gaussian posteriors, see
    {altar.bayesian.samplers.Perturbation}
    """

    # protocol-shaped obligations (called by the shim)
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me and my parts given an {application} context
        """
        self.info = application.info
        self.warning = application.warning
        self.error = application.error
        self.rng = application.rng.rng
        self.stepcounter.initialize(application=application)
        return self


    def sample_posterior(self, annealer: Annealer, step: CoolingStep) -> Statistics:
        """
        Draw the samples of {step} from the posterior at its β
        """
        dispatcher = annealer.dispatcher
        dispatcher.notify(event=dispatcher.sample_posterior_start, controller=annealer)
        model = annealer.model
        posterior = self.posterior(model=model, beta=step.beta)
        theta = self.host_theta(step=step)
        samples = theta.shape[0]
        if posterior.bounded:
            # trajectories from the current samples, which are in the box
            theta[...] = numpy.clip(theta, posterior.low, posterior.high)
            trajectories = bounces = 0
            for _ in walk(self.stepcounter, theta=theta, beta=step.beta, annealer=annealer):
                bounces += posterior.trajectory(theta=theta, time=self.trajectory_time,
                                                rng=self.rng, max_bounces=self.max_bounces)
                trajectories += 1
            self.info.log(f"perturbation: {trajectories} trajectories, "
                          f"{bounces / max(1, trajectories * samples):.3g} bounces per trajectory")
        else:
            # independent draws, one is enough
            trajectories = 1
            theta[...] = posterior.mean + posterior.perturbation(rows=samples, rng=self.rng)
        self.store(annealer=annealer, step=step, theta=theta)
        # the evidence, at β = 1
        if step.beta == 1:
            self.evidence(annealer=annealer, step=step, posterior=posterior, theta=theta)
        dispatcher.notify(event=dispatcher.sample_posterior_finish, controller=annealer)
        return Statistics(trajectories * samples, 0, 0)


    def update(self, annealer: Annealer, statistics: Statistics) -> None:
        """
        Nothing to adapt
        """
        return


    def posterior(self, model: typing.Any, beta: float) -> Posterior:
        """
        The posterior of {model} at {beta}: from its whitened linear system and its gaussian prior,
        or, on the matrix-free path ({solver}), from its {linear_operator}
        """
        if self.solver != "dense":
            try:
                operator = model.linear_operator(beta=beta)
            except (AttributeError, NotImplementedError):
                operator = None
            if operator is None and self.solver == "operator":
                # the dense system of the model, solved matrix-free: for the checks
                try:
                    Gw, dw, _ = model.linear_system()
                    mu, precision, low, high = model.gaussian_prior()
                except NotImplementedError as error:
                    self.error.log(f"perturbation: {error}")
                    raise SystemExit(1)
                if numpy.isfinite(low).any() or numpy.isfinite(high).any():
                    self.error.log("perturbation: the matrix-free path doesn't sample priors "
                                   "with bounds")
                    raise SystemExit(1)
                operator = DenseOperator(Gw=Gw, dw=dw, mu=mu, precision=precision, beta=beta)
            if operator is not None:
                try:
                    posterior = OperatorPosterior(operator=operator, tolerance=self.tolerance,
                                                  max_iterations=self.max_iterations,
                                                  info=self.info)
                except ValueError as error:
                    self.error.log(f"perturbation: {error}")
                    raise SystemExit(1)
                if self.walks == 0:
                    self.info.log(f"perturbation: {operator.size} parameters, matrix-free "
                                  f"({type(operator).__name__})")
                self.walks += 1
                return posterior
        try:
            Gw, dw, _ = model.linear_system()
            mu, precision, low, high = model.gaussian_prior()
            posterior = Posterior(Gw=Gw, dw=dw, mu=mu, precision=precision, low=low, high=high,
                                  beta=beta)
        except (NotImplementedError, ValueError) as error:
            self.error.log(f"perturbation: {error}")
            raise SystemExit(1)
        if self.walks == 0:
            walls = int(numpy.isfinite(low).sum() + numpy.isfinite(high).sum())
            self.info.log(f"perturbation: {mu.size} parameters, {dw.size} observations, "
                          f"{walls} walls")
        self.walks += 1
        return posterior


    def evidence(self, annealer: Annealer, step: CoolingStep, posterior: Posterior,
                 theta: numpy.ndarray) -> None:
        """
        log p(d) = log p(θ) + log p(d|θ) - log p(θ|d) at each sample, exact for a gaussian on a
        box, its mass by GHK; the spread over the samples checks that the model is one
        """
        if posterior.log_density is None:
            self.info.log("perturbation: no log evidence on the matrix-free path")
            return
        if posterior.bounded and self.evidence_draws == 0:
            return
        prior, data = (numpy.asarray(likelihood, dtype=float) for likelihood in self.host_likelihoods(step))
        log_mass, error = posterior.log_mass(draws=self.evidence_draws)
        estimates = prior + data - posterior.log_density(theta) + log_mass
        value = float(numpy.median(estimates))
        spread = float(numpy.abs(estimates - value).max())
        self.info.log(f"perturbation: log evidence {value:.6g}, spread {spread:.2g} over the samples"
                      + (f", box mass exp({log_mass:.6g} ± {error:.2g})" if posterior.bounded else ""))
        if spread > 1e-2 + 1e-4 * abs(value):
            self.warning.log(f"perturbation: the posterior of the model is not the gaussian i "
                             f"sample: log p(θ) + log p(d|θ) - log p(θ|d) varies by {spread:.3g} "
                             f"over the samples")
        # for the archive, unless an annealing schedule estimates it
        if isinstance(annealer.scheduler, ConstantTemperature):
            annealer.scheduler.log_evidence = value
        return


    # backend hooks
    def host_theta(self, step: CoolingStep) -> numpy.ndarray:
        """
        The physical samples of {step}, to move in place
        """
        return step.theta


    def store(self, annealer: Annealer, step: CoolingStep, theta: numpy.ndarray) -> None:
        """
        Put the samples back in {step}, with their likelihoods
        """
        step.refresh_sampling(model=annealer.model)
        for likelihood in (step.prior, step.data, step.posterior):
            likelihood[...] = 0
        annealer.model.likelihoods(annealer=annealer, step=step)
        return


    def host_likelihoods(self, step: CoolingStep) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        The log prior and log likelihood of the samples of {step}
        """
        return step.prior, step.data


    # private data; set by the shim
    stepcounter: StepCounter
    trajectory_time: float
    max_bounces: int
    evidence_draws: int
    solver: str
    tolerance: float
    max_iterations: int
    walks: int = 0
    info: journal.info
    warning: journal.warning
    error: journal.error
    rng: numpy.random.Generator


# end of file
