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
        The posterior of {model} at {beta}, from its whitened linear system and its gaussian prior
        """
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
    walks: int = 0
    info: journal.info
    warning: journal.warning
    error: journal.error
    rng: numpy.random.Generator


# end of file
