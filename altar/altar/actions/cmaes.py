# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import numpy


# declaration
class CMAES:
    """
    The covariance matrix adaptation evolution strategy, (mu/mu_w, lambda)-CMA-ES, minimizing
    a function by sampling populations from N(m, sigma^2 C) and adapting m, sigma and C to the
    best of each; after Hansen, "The CMA evolution strategy: a tutorial" (2016).

    Bounds are handled by repair: the points outside are moved to the nearest bound, evaluated
    there, with a penalty on the distance moved, and the repaired points update the
    distribution, so that optima on a bound are reached
    """


    def __init__(self, x0, scale, lower=None, upper=None, population=None, seed=None):
        """
        Start at {x0}, with the initial standard deviations {scale} of each coordinate
        """
        self.mean = numpy.array(x0, dtype=float)
        n = self.dimension = self.mean.size
        self.scale = numpy.broadcast_to(numpy.asarray(scale, dtype=float), (n,)).copy()
        self.lower = numpy.full(n, -numpy.inf) if lower is None else numpy.asarray(lower, float)
        self.upper = numpy.full(n, numpy.inf) if upper is None else numpy.asarray(upper, float)
        self.mean = numpy.clip(self.mean, self.lower, self.upper)
        self.rng = numpy.random.default_rng(seed)

        # the population, the parents and their weights
        self.population = population or 4 + int(3 * numpy.log(n))
        self.parents = self.population // 2
        w = numpy.log(self.parents + 0.5) - numpy.log(numpy.arange(1, self.parents + 1))
        self.weights = w / w.sum()
        self.mueff = 1 / (self.weights ** 2).sum()

        # the learning rates and the damping, as in the tutorial
        mueff = self.mueff
        self.cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        self.cs = (mueff + 2) / (n + mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + mueff)
        self.cmu = min(1 - self.c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        self.damps = 1 + 2 * max(0, numpy.sqrt((mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = numpy.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # the state: sigma = 1 and C = diag(scale^2), the evolution paths, and the best so far
        self.sigma = 1.0
        self.C = numpy.diag(self.scale ** 2)
        self.B, self.D = numpy.eye(n), self.scale.copy()
        self.pc, self.ps = numpy.zeros(n), numpy.zeros(n)
        self.generation = 0
        self.best = (numpy.inf, self.mean.copy())
        self.history = []
        return


    def ask(self):
        """
        Sample a population: the points to evaluate, inside the bounds, and the penalty of each
        for the distance its repair moved it
        """
        z = self.rng.standard_normal((self.population, self.dimension))
        x = self.mean + self.sigma * (z * self.D) @ self.B.T
        repaired = numpy.clip(x, self.lower, self.upper)
        penalty = (((x - repaired) / self.scale) ** 2).sum(axis=1)
        return repaired, penalty


    def tell(self, x, values, penalty=None):
        """
        Update the distribution from the points {x} and their {values}, plus a {penalty}
        weighted to the spread of the values
        """
        values = numpy.asarray(values, dtype=float)
        f = values.copy()
        if penalty is not None and numpy.any(penalty > 0):
            spread = numpy.ptp(values[numpy.isfinite(values)]) if numpy.isfinite(values).any() else 1
            f = f + (spread + 1) * penalty
        f[~numpy.isfinite(f)] = numpy.inf
        order = numpy.argsort(f)
        if values[order[0]] < self.best[0]:
            self.best = (float(values[order[0]]), x[order[0]].copy())

        n, old = self.dimension, self.mean
        chosen = x[order[:self.parents]]
        self.mean = self.weights @ chosen
        y = (self.mean - old) / self.sigma
        # the evolution paths
        invsqrt = (self.B / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + numpy.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrt @ y
        decay = 1 - (1 - self.cs) ** (2 * (self.generation + 1))
        hsig = numpy.linalg.norm(self.ps) / numpy.sqrt(decay) / self.chiN < 1.4 + 2 / (n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * numpy.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        # the covariance: the rank-one and the rank-mu updates
        steps = (chosen - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (numpy.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (steps.T * self.weights) @ steps)
        # the step size
        self.sigma *= numpy.exp((self.cs / self.damps) * (numpy.linalg.norm(self.ps) / self.chiN - 1))
        # the new axes and scales
        self.C = (self.C + self.C.T) / 2
        eigenvalues, self.B = numpy.linalg.eigh(self.C)
        self.D = numpy.sqrt(numpy.maximum(eigenvalues, 1e-300))

        self.generation += 1
        good = numpy.isfinite(values)
        self.history.append((float(values[order[0]]),
                             float(numpy.median(values[good])) if good.any() else numpy.inf,
                             float(self.sigma * self.D.max())))
        return


    def spread(self):
        """
        The current standard deviation of each coordinate, relative to its initial one
        """
        return self.sigma * numpy.sqrt(numpy.diag(self.C)) / self.scale


    def stop(self, tolx=1e-6, tolfun=1e-6, window=None):
        """
        The reason to stop, if any: every coordinate has shrunk below {tolx} of its start, the best
        values of the last {window} generations differ by less than {tolfun}, or C has degenerated
        """
        window = window or 10 + int(numpy.ceil(30 * self.dimension / self.population))
        if self.generation and self.spread().max() < tolx:
            return f"the step sizes fell below {tolx:g} of their start"
        if len(self.history) >= window:
            recent = [h[0] for h in self.history[-window:]]
            if max(recent) - min(recent) < tolfun:
                return f"the best value changed less than {tolfun:g} in {window} generations"
        # the conditioning of C in units of the initial scales, so that it is independent of them
        normalized = numpy.linalg.eigvalsh(self.C / numpy.outer(self.scale, self.scale))
        if normalized.max() > 1e14 * max(normalized.min(), 1e-300):
            return "the covariance became ill-conditioned"
        return None


# end of file
