# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import pathlib
import numpy
# get the package
import altar
# the samples and the misfit, as the forward check reads and measures them
from .Forward import Misfit, load_samples
# the optimizers
from .cmaes import CMAES
from .minimizers import LevenbergMarquardt, LBFGS, Scaled


# declaration
class Optimize(altar.panel(), family='altar.actions.optimize'):
    """
    The best fit of a model to its data: maximize the data log likelihood, or, with {posterior},
    the posterior under the conjugate prior, over the parameters inside the supports of their
    priors, by CMA-ES, which needs no gradient, Levenberg-Marquardt or L-BFGS, which use a jacobian
    by finite differences; each step runs the forward model on a batch of points, shared among
    the processes of an mpi run
    """


    # user configurable state
    method = altar.properties.str(default="cma")
    method.validators = altar.constraints.isMember("cma", "lm", "lbfgs")
    method.doc = "the optimizer: 'cma', the covariance matrix adaptation evolution strategy; " \
                 "'lm', Levenberg-Marquardt; 'lbfgs', projected L-BFGS"

    theta = altar.properties.path(default=None)
    theta.doc = "the start: an archived step (.h5 with ParameterSets), or a .txt/.h5 file with " \
                "one sample per row, whose mean is used; default the means of the priors"

    dataset = altar.properties.str(default=None)
    dataset.doc = "the dataset holding {theta} in a plain .h5 file; default the first one"

    posterior = altar.properties.bool(default=False)
    posterior.doc = "add the conjugate prior to the objective, for the maximum of the " \
                    "posterior instead of the best fit to the data"

    population = altar.properties.int(default=None)
    population.doc = "cma: the points of each generation; default 4 + 3 ln(parameters); larger " \
                     "populations search more globally, at the same cost per generation on a gpu"

    generations = altar.properties.int(default=2000)
    generations.doc = "cma: the most generations to run"

    iterations = altar.properties.int(default=200)
    iterations.doc = "lm, lbfgs: the most iterations to run"

    step = altar.properties.float(default=0.3)
    step.doc = "the initial standard deviation of each parameter, relative to the spread of " \
               "the samples in {theta}, or else to the standard deviation of its prior; the " \
               "unit in which lm and lbfgs measure their steps"

    difference = altar.properties.float(default=1e-3)
    difference.doc = "lm, lbfgs: the finite-difference step of each parameter, relative to " \
                     "its {step}"

    memory = altar.properties.int(default=10)
    memory.doc = "lbfgs: the number of curvature pairs to keep"

    tolerance = altar.properties.float(default=1e-6)
    tolerance.doc = "stop when the best objective changes less than this over a window of " \
                    "generations for cma, of 5 iterations for lm and lbfgs"

    seed = altar.properties.int(default=0)
    seed.doc = "cma: the seed of the random populations"

    batch = altar.properties.int(default=None)
    batch.doc = "run at most this many points at once in each process; default all of its share"

    output = altar.properties.path(default="optimize.h5")
    output.doc = "the .h5 file for the best point and the history; the best point is also " \
                 "written as a one-row .txt next to it, for the other actions' {theta}"


    # commands
    @altar.export(tip="find the best fit of the model to its data")
    def default(self, plexus, **kwds):
        """
        Optimize from the start, report the best point by parameter set and save it
        """
        model = plexus.model
        lower, upper = self.bounds(model=model)
        prior = self.prior(model=model)
        if self.posterior and prior is None:
            raise ValueError("'posterior' needs the conjugate prior of the model")
        start, scale, sampled = self.start(model=model, lower=lower, upper=upper, prior=prior)
        problem = Problem(plexus=plexus, model=model, start=start, posterior=self.posterior,
                          prior=prior, batch=self.batch,
                          steps=self.difference * scale, lower=lower, upper=upper)
        # the workers of an mpi run only evaluate the points the manager sends them
        if not problem.manager:
            problem.serve()
            return 0

        channel = plexus.info
        what = "log posterior (conjugate prior)" if self.posterior else "data log likelihood"
        name = {"cma": "CMA-ES", "lm": "Levenberg-Marquardt", "lbfgs": "L-BFGS"}[self.method]
        channel.line(f"optimize: {name} on the {what}, {model.parameters} parameters,"
                     f" {problem.processes} process(es) -> '{self.output}'")
        channel.line(f"  start: {'the prior means' if self.theta is None else self.theta},"
                     f" initial sd {self.step:g} x {'the sample' if sampled else 'the prior'} sd")
        try:
            initial = float(problem.values(start[None, :])[0])
            channel.line(f"  start {what} {initial:.8g}")
            run = {"cma": self.cma, "lm": self.descent, "lbfgs": self.descent}[self.method]
            result = run(problem=problem, start=start, scale=scale, lower=lower, upper=upper,
                         initial=initial, channel=channel)
            best, x = result["best"], result["x"]
            likelihood = float(problem.values(x[None, :], with_prior=False)[0]) \
                if self.posterior else best
        finally:
            problem.release()

        channel.line(f"  stopped after {result['steps']} {result['unit']},"
                     f" {problem.evaluations} evaluations: {result['reason']}")
        channel.line(f"  best {what} {best:.8g}, {best - initial:+.6g} from the start")
        if self.posterior:
            channel.line(f"  its data log likelihood {likelihood:.8g}")
        self.report(model=model, x=x, lower=lower, upper=upper, sd=result.get("sd"),
                    channel=channel)
        self.save(result=result, start=start, initial=initial, likelihood=likelihood,
                  lower=lower, upper=upper)
        channel.log()
        return 0


    # the optimizers
    def cma(self, problem, start, scale, lower, upper, initial, channel):
        """
        CMA-ES from {start}, with the initial standard deviations {scale}; the start, of value
        {initial}, is the best point until a better one is found
        """
        es = CMAES(x0=start, scale=scale, lower=lower, upper=upper, population=self.population,
                   seed=self.seed)
        es.best = (-initial, start.copy())
        channel.line(f"  population {es.population}")
        every = max(1, self.generations // 20)
        reason = f"{self.generations} generations"
        while es.generation < self.generations:
            x, penalty = es.ask()
            es.tell(x, -problem.values(x), penalty)
            if es.generation % every == 0:
                best, median, _ = es.history[-1]
                channel.line(f"  generation {es.generation:5d}: best {-es.best[0]:.8g},"
                             f" this generation {-best:.8g}, median {-median:.8g},"
                             f" max sd {es.spread().max():.3g} x start")
            done = es.stop(tolx=self.tolerance, tolfun=self.tolerance)
            if done:
                reason = done
                break
        history = numpy.array(es.history)
        return {"best": -es.best[0], "x": es.best[1], "steps": es.generation, "unit": "generations",
                "reason": reason, "history": {"best": -history[:, 0], "median": -history[:, 1],
                                              "sd": history[:, 2]},
                "final": {"mean": es.mean, "covariance": es.sigma ** 2 * es.C},
                "attributes": {"population": es.population}}


    def descent(self, problem, start, scale, lower, upper, initial, channel):
        """
        Levenberg-Marquardt or L-BFGS from {start}, in units of {scale}
        """
        scaled = Scaled(problem=problem, origin=start, scale=scale)
        u0, ulow, uhigh = numpy.zeros_like(start), scaled.to(lower), scaled.to(upper)
        if self.method == "lm":
            m = LevenbergMarquardt(problem=scaled, x0=u0, lower=ulow, upper=uhigh)
        else:
            m = LBFGS(problem=scaled, x0=u0, lower=ulow, upper=uhigh, memory=self.memory)
        every = max(1, self.iterations // 20)
        reason = f"{self.iterations} iterations"
        window = 5
        while m.iteration < self.iterations:
            improved = m.step()
            if m.iteration % every == 0:
                channel.line(f"  iteration {m.iteration:5d}: {problem.constant - m.value:.8g}")
            if not improved:
                reason = "no step improved the fit"
                break
            if len(m.history) > window and m.history[-window - 1] - m.value < self.tolerance:
                reason = f"the objective changed less than {self.tolerance:g} in {window} iterations"
                break
        x = scaled.back(m.x)
        result = {"best": problem.constant - m.value, "x": x, "steps": m.iteration,
                  "unit": "iterations", "reason": reason,
                  "history": {"best": problem.constant - numpy.array(m.history)}}
        if self.method == "lm":
            # the covariance at the optimum, (J^T J)^{-1}, in the units of the parameters
            C = scale[:, None] * m.covariance() * scale[None, :]
            result["final"] = {"mean": x, "covariance": C}
            result["sd"] = numpy.sqrt(numpy.clip(numpy.diag(C), 0, None))
        return result


    # implementation details
    def bounds(self, model):
        """
        The lower and upper bounds of each parameter, from the supports of their priors
        """
        lower = numpy.full(model.parameters, -numpy.inf)
        upper = numpy.full(model.parameters, numpy.inf)
        for name in model.psets_list:
            pset = model.psets[name]
            support = getattr(pset.prior, "support", None)
            if support is not None:
                columns = slice(pset.offset, pset.offset + pset.count)
                lower[columns], upper[columns] = support[0], support[1]
        return lower, upper


    def prior(self, model):
        """
        The mean and the variance of each parameter under the conjugate prior, if the model has one
        """
        try:
            return model.conjugate_prior()
        except (AttributeError, NotImplementedError):
            return None


    def start(self, model, lower, upper, prior):
        """
        The starting point, the initial standard deviation of each parameter, and whether it
        comes from the spread of samples
        """
        sampled = False
        bounded = numpy.isfinite(lower) & numpy.isfinite(upper)
        if prior is not None:
            mean, sd = prior[0].copy(), numpy.sqrt(prior[1])
        else:
            mean = numpy.where(bounded, (lower + upper) / 2, 0.0)
            sd = numpy.where(bounded, (upper - lower) / numpy.sqrt(12), 1.0)
        if self.theta is not None:
            θ = load_samples(model=model, path=self.theta, dataset=self.dataset)
            mean = θ.mean(axis=0)
            if θ.shape[0] > 1 and (θ.std(axis=0) > 0).all():
                sd, sampled = θ.std(axis=0), True
        scale = self.step * sd
        # no wider than a quarter of a finite support
        scale = numpy.where(bounded, numpy.minimum(scale, (upper - lower) / 4), scale)
        return numpy.clip(mean, lower, upper), scale, sampled


    def report(self, model, x, lower, upper, channel, sd=None):
        """
        Summarize the best point by parameter set on {channel}, with the parameters on a bound
        and, from lm, the median standard deviation at the optimum
        """
        channel.line(f"  {'parameter set':16s}{'count':>6s}{'mean':>14s}{'[min, max]':>30s}"
                     f"{'on a bound':>12s}" + (f"{'median sd':>12s}" if sd is not None else ""))
        for name in model.psets_list:
            pset = model.psets[name]
            columns = slice(pset.offset, pset.offset + pset.count)
            v, lo, hi = x[columns], lower[columns], upper[columns]
            width = numpy.where(numpy.isfinite(hi - lo), hi - lo, 1.0)
            edge = int(((v - lo <= 1e-6 * width) | (hi - v <= 1e-6 * width)).sum())
            channel.line(f"  {name:16s}{pset.count:6d}{v.mean():14.6g}"
                         f"{f'[{v.min():.6g}, {v.max():.6g}]':>30s}{edge:12d}"
                         + (f"{numpy.median(sd[columns]):12.4g}" if sd is not None else ""))
        if sd is not None:
            channel.line("  (sd: from (J^T J)^{-1} at the optimum; zero for the parameters held"
                         " on a bound)")
        return


    def save(self, result, start, initial, likelihood, lower, upper):
        """
        Save the best point, the start, the final state of the optimizer and the history
        """
        import h5py
        x = result["x"]
        with h5py.File(str(self.output), "w") as h5:
            h5["best/theta"], h5["best/objective"] = x, result["best"]
            h5["best/loglikelihood"] = likelihood
            h5["start/theta"], h5["start/objective"] = start, initial
            h5["bounds/lower"], h5["bounds/upper"] = lower, upper
            for key, value in result["history"].items():
                h5[f"history/{key}"] = value
            for key, value in result.get("final", {}).items():
                h5[f"final/{key}"] = value
            if "sd" in result:
                h5["best/sd"] = result["sd"]
            h5.attrs["method"], h5.attrs["posterior"] = self.method, bool(self.posterior)
            h5.attrs["steps"] = result["steps"]
            for key, value in result.get("attributes", {}).items():
                h5.attrs[key] = value
        numpy.savetxt(pathlib.Path(str(self.output)).with_suffix(".txt"), x[None, :])
        return


class Problem:
    """
    The fit of a model to its data, as least squares: the whitened residuals of the rows of theta,
    summed over the models of an ensemble, each on its own columns, plus the conjugate prior if
    asked for, and their jacobian by central differences. The rows of each batch are shared among
    the processes of an mpi run: the manager sends them, every process runs its part, and the
    manager collects the residuals
    """


    def __init__(self, plexus, model, start, posterior, prior, batch, steps, lower, upper):
        import mpi
        self.plexus, self.posterior, self.prior, self.batch = plexus, posterior, prior, batch
        self.steps, self.lower, self.upper = steps, lower, upper
        self.world = mpi.world
        self.processes = self.world.size
        self.manager = self.world.rank == 0
        self.evaluations = 0
        members = getattr(model, "models", None)
        self.parts = []
        for member, columns in ([(member, model.columns(name)) for name, member in members.items()]
                                if members else [(model, numpy.arange(model.parameters))]):
            # C_p, if any, at the start, as the forward check does at the posterior mean
            cp = getattr(member, "cp", None)
            if cp is not None:
                cp.apply(model=member, theta=start[columns])
            self.parts.append((member, columns, Misfit(dataobs=member.dataobs)))
        # the normalization: the objective is the constant minus half the sum of squares
        self.constant = sum(misfit.normalization for _, _, misfit in self.parts)
        return


    def residuals(self, theta, with_prior=None):
        """
        The residuals of each row of {theta}, on the manager; a collective call
        """
        with_prior = self.posterior if with_prior is None else with_prior
        theta = self.world.bcast(item=("residuals", theta, with_prior), source=0)[1]
        return self.share(theta=theta, with_prior=with_prior)


    def share(self, theta, with_prior):
        """
        Run my share of the rows of {theta} and collect them on the manager
        """
        rows = numpy.array_split(numpy.arange(theta.shape[0]), self.processes)[self.world.rank]
        mine = self.evaluate(theta=theta[rows], with_prior=with_prior)
        parts = self.world.gatherObject(item=mine, root=0)
        if self.manager:
            self.evaluations += theta.shape[0]
            return numpy.concatenate([part for part in parts if part.shape[0]] or parts, axis=0)
        return None


    def evaluate(self, theta, with_prior):
        """
        The whitened residuals of the rows of {theta}, in batches
        """
        blocks = []
        size = self.batch or max(theta.shape[0], 1)
        for begin in range(0, theta.shape[0], size):
            rows = theta[begin:begin + size]
            columns = []
            for member, cols, misfit in self.parts:
                p = member.forward_problem(application=self.plexus, theta=rows[:, cols])["data"]
                columns.append(misfit.whiten(p))
            if with_prior:
                mean, variance = self.prior
                columns.append((rows - mean) / numpy.sqrt(variance))
            blocks.append(numpy.hstack(columns))
        if not blocks:
            width = sum(misfit.observed.size for _, _, misfit in self.parts)
            return numpy.zeros((0, width + (theta.shape[1] if with_prior else 0)))
        return numpy.vstack(blocks)


    def values(self, theta, with_prior=None):
        """
        The objective of each row of {theta}: the constant minus half the sum of squares
        """
        R = self.residuals(theta=theta, with_prior=with_prior)
        return self.constant - 0.5 * (R ** 2).sum(axis=1)


    def jacobian(self, x):
        """
        The jacobian of the residuals at {x}, by central differences inside the bounds, all the
        perturbed points in one batch
        """
        n = x.size
        plus = numpy.minimum(x + self.steps, self.upper)
        minus = numpy.maximum(x - self.steps, self.lower)
        X = numpy.vstack([numpy.where(numpy.eye(n, dtype=bool), plus[None, :], x[None, :]),
                          numpy.where(numpy.eye(n, dtype=bool), minus[None, :], x[None, :])])
        R = self.residuals(X)
        J = (R[:n] - R[n:]).T / (plus - minus)
        # a derivative from a forward run that failed is unknown: leave it out
        J[~numpy.isfinite(J)] = 0
        return J


    def serve(self):
        """
        On a worker: evaluate the shares of the manager's batches until it is done
        """
        while True:
            command, theta, with_prior = self.world.bcast(item=None, source=0)
            if command != "residuals":
                return
            self.share(theta=theta, with_prior=with_prior)


    def release(self):
        """
        On the manager: tell the workers there is nothing more to evaluate
        """
        if self.processes > 1:
            self.world.bcast(item=("done", None, None), source=0)
        return


# end of file
