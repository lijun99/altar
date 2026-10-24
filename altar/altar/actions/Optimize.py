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
# the optimizer
from .cmaes import CMAES


# declaration
class Optimize(altar.panel(), family='altar.actions.optimize'):
    """
    The best fit of a model to its data: maximize the data log likelihood, or, with {posterior},
    the posterior under the conjugate prior, over the parameters inside the supports of their
    priors, with the covariance matrix adaptation evolution strategy (CMA-ES). It needs no
    gradient; each generation is one batch of the forward model
    """


    # user configurable state
    theta = altar.properties.path(default=None)
    theta.doc = "the start: an archived step (.h5 with ParameterSets), or a .txt/.h5 file with " \
                "one sample per row, whose mean is used; default the means of the priors"

    dataset = altar.properties.str(default=None)
    dataset.doc = "the dataset holding {theta} in a plain .h5 file; default the first one"

    posterior = altar.properties.bool(default=False)
    posterior.doc = "add the conjugate prior to the objective, for the maximum of the " \
                    "posterior instead of the best fit to the data"

    population = altar.properties.int(default=None)
    population.doc = "the points of each generation; default 4 + 3 ln(parameters); larger " \
                     "populations search more globally, at the same cost per generation on a gpu"

    generations = altar.properties.int(default=2000)
    generations.doc = "the most generations to run"

    step = altar.properties.float(default=0.3)
    step.doc = "the initial standard deviation of each parameter, relative to the spread of " \
               "the samples in {theta}, or else to the standard deviation of its prior"

    tolerance = altar.properties.float(default=1e-6)
    tolerance.doc = "stop when the best objective changes less than this over a window of " \
                    "generations, or every step size falls below this fraction of its start"

    seed = altar.properties.int(default=0)
    seed.doc = "the seed of the random populations"

    batch = altar.properties.int(default=None)
    batch.doc = "evaluate at most this many points at once; default the whole population"

    output = altar.properties.path(default="optimize.h5")
    output.doc = "the .h5 file for the best point and the history; the best point is also " \
                 "written as a one-row .txt next to it, for the other actions' {theta}"


    # commands
    @altar.export(tip="find the best fit of the model to its data")
    def default(self, plexus, **kwds):
        """
        Run CMA-ES from the start, report the best point by parameter set and save it
        """
        model = plexus.model
        channel = plexus.info
        lower, upper = self.bounds(model=model)
        prior = self.prior(model=model)
        if self.posterior and prior is None:
            raise ValueError("'posterior' needs the conjugate prior of the model")
        start, scale, sampled = self.start(model=model, lower=lower, upper=upper, prior=prior)
        objective = self.objective(plexus=plexus, model=model, start=start, prior=prior)

        what = "log posterior (conjugate prior)" if self.posterior else "data log likelihood"
        channel.line(f"optimize: CMA-ES on the {what}, {model.parameters} parameters"
                     f" -> '{self.output}'")
        channel.line(f"  start: {'the prior means' if self.theta is None else self.theta},"
                     f" initial sd {self.step:g} x {'the sample' if sampled else 'the prior'} sd")
        initial = float(objective(start[None, :])[0])
        es = CMAES(x0=start, scale=scale, lower=lower, upper=upper, population=self.population,
                   seed=self.seed)
        channel.line(f"  population {es.population}, start {what} {initial:.8g}")

        every = max(1, self.generations // 20)
        reason = f"{self.generations} generations"
        while es.generation < self.generations:
            x, penalty = es.ask()
            es.tell(x, -objective(x), penalty)
            if es.generation % every == 0:
                best, median, _ = es.history[-1]
                channel.line(f"  generation {es.generation:5d}: best {-es.best[0]:.8g},"
                             f" this generation {-best:.8g}, median {-median:.8g},"
                             f" max sd {es.spread().max():.3g} x start")
            done = es.stop(tolx=self.tolerance, tolfun=self.tolerance)
            if done:
                reason = done
                break

        best, x = -es.best[0], es.best[1]
        likelihood = float(objective(x[None, :], with_prior=False)[0]) if self.posterior else best
        channel.line(f"  stopped after {es.generation} generations,"
                     f" {es.generation * es.population} evaluations: {reason}")
        channel.line(f"  best {what} {best:.8g}, {best - initial:+.6g} from the start")
        if self.posterior:
            channel.line(f"  its data log likelihood {likelihood:.8g}")
        self.report(model=model, x=x, lower=lower, upper=upper, channel=channel)
        self.save(es=es, start=start, initial=initial, best=best, likelihood=likelihood, x=x,
                  lower=lower, upper=upper)
        channel.log()
        return 0


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


    def objective(self, plexus, model, start, prior):
        """
        The function to maximize: the data log likelihood of the rows of theta, summed over the
        models of an ensemble, each on its own columns, plus the conjugate prior if asked for
        """
        members = getattr(model, "models", None)
        parts = []
        for member, columns in ([(member, model.columns(name)) for name, member in members.items()]
                                if members else [(model, numpy.arange(model.parameters))]):
            # C_p, if any, at the start, as the forward check does at the posterior mean
            cp = getattr(member, "cp", None)
            if cp is not None:
                cp.apply(model=member, theta=start[columns])
            parts.append((member, columns, Misfit(dataobs=member.dataobs)))

        def evaluate(theta, with_prior=self.posterior):
            values = numpy.empty(theta.shape[0])
            size = self.batch or theta.shape[0]
            for begin in range(0, theta.shape[0], size):
                rows = theta[begin:begin + size]
                total = numpy.zeros(rows.shape[0])
                for member, columns, misfit in parts:
                    p = member.forward_problem(application=plexus, theta=rows[:, columns])["data"]
                    total += misfit.normalization - 0.5 * (misfit.whiten(p) ** 2).sum(axis=1)
                if with_prior:
                    mean, variance = prior
                    total -= 0.5 * (((rows - mean) ** 2) / variance).sum(axis=1)
                values[begin:begin + size] = total
            return values

        return evaluate


    def report(self, model, x, lower, upper, channel):
        """
        Summarize the best point by parameter set on {channel}, with the parameters on a bound
        """
        channel.line(f"  {'parameter set':16s}{'count':>6s}{'mean':>14s}{'[min, max]':>30s}"
                     f"{'on a bound':>12s}")
        for name in model.psets_list:
            pset = model.psets[name]
            columns = slice(pset.offset, pset.offset + pset.count)
            v, lo, hi = x[columns], lower[columns], upper[columns]
            width = numpy.where(numpy.isfinite(hi - lo), hi - lo, 1.0)
            edge = int(((v - lo <= 1e-6 * width) | (hi - v <= 1e-6 * width)).sum())
            channel.line(f"  {name:16s}{pset.count:6d}{v.mean():14.6g}"
                         f"{f'[{v.min():.6g}, {v.max():.6g}]':>30s}{edge:12d}")
        return


    def save(self, es, start, initial, best, likelihood, x, lower, upper):
        """
        Save the best point, the start, the final distribution and the history
        """
        import h5py
        history = numpy.array(es.history)
        with h5py.File(str(self.output), "w") as h5:
            h5["best/theta"], h5["best/objective"] = x, best
            h5["best/loglikelihood"] = likelihood
            h5["start/theta"], h5["start/objective"] = start, initial
            h5["final/mean"], h5["final/covariance"] = es.mean, es.sigma ** 2 * es.C
            h5["bounds/lower"], h5["bounds/upper"] = lower, upper
            h5["history/best"], h5["history/median"] = -history[:, 0], -history[:, 1]
            h5["history/sd"] = history[:, 2]
            h5.attrs["method"], h5.attrs["posterior"] = "cma-es", bool(self.posterior)
            h5.attrs["population"], h5.attrs["generations"] = es.population, es.generation
        numpy.savetxt(pathlib.Path(str(self.output)).with_suffix(".txt"), x[None, :])
        return


# end of file
