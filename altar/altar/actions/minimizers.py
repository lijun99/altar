# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Bounded least squares, minimizing |r(x)|^2 / 2 for the residuals r of a {problem}: Levenberg-
Marquardt and projected L-BFGS. The problem provides {residuals(X)}, the residuals of each row of
{X} as the rows of a matrix, and {jacobian(x)}, the jacobian of r at {x}; both minimizers try a
few points at once in each step, as one batch of the forward model
"""

# externals
import numpy


def cost(R):
    """
    |r|^2 / 2 of each row of {R}
    """
    return 0.5 * (R ** 2).sum(axis=1)


def free(x, gradient, lower, upper):
    """
    The parameters not held on a bound by the gradient pushing them out of it
    """
    width = numpy.where(numpy.isfinite(upper - lower), upper - lower, 1.0)
    at_lower = (x - lower <= 1e-12 * width) & (gradient > 0)
    at_upper = (upper - x <= 1e-12 * width) & (gradient < 0)
    return ~(at_lower | at_upper)


class Scaled:
    """
    A {problem} in units of the {scale} of each parameter, about {origin}: x = origin + scale u
    """


    def __init__(self, problem, origin, scale):
        self.problem, self.origin, self.scale = problem, origin, scale
        return


    def residuals(self, U):
        return self.problem.residuals(self.origin + self.scale * U)


    def jacobian(self, u):
        return self.problem.jacobian(self.origin + self.scale * u) * self.scale


    def to(self, x):
        """
        The scaled coordinates of {x}
        """
        return (x - self.origin) / self.scale


    def back(self, u):
        """
        The parameters at the scaled coordinates {u}
        """
        return self.origin + self.scale * u


class LevenbergMarquardt:
    """
    Levenberg-Marquardt with Marquardt's scaling of the damping, (J^T J + λ diag J^T J) δ = -J^T r,
    on the parameters not held on a bound; each iteration tries a few dampings at once
    """


    def __init__(self, problem, x0, lower, upper, damping=1e-3, trials=(0.1, 1, 10, 100)):
        self.problem = problem
        self.lower, self.upper = numpy.asarray(lower, float), numpy.asarray(upper, float)
        self.x = numpy.clip(numpy.asarray(x0, float), self.lower, self.upper)
        self.damping, self.trials = damping, numpy.asarray(trials, float)
        self.r = problem.residuals(self.x[None, :])[0]
        self.value = float(cost(self.r[None, :])[0])
        self.iteration = 0
        self.history = [self.value]
        return


    def step(self):
        """
        One iteration; False if no damping improved the fit
        """
        J = self.problem.jacobian(self.x)
        A, g = J.T @ J, J.T @ self.r
        active = free(self.x, g, self.lower, self.upper)
        candidates = []
        for λ in self.damping * self.trials:
            δ = numpy.zeros_like(self.x)
            Af = A[numpy.ix_(active, active)]
            δ[active] = numpy.linalg.solve(Af + λ * numpy.diag(numpy.maximum(numpy.diag(Af), 1e-300)),
                                           -g[active])
            candidates.append(numpy.clip(self.x + δ, self.lower, self.upper))
        X = numpy.array(candidates)
        R = self.problem.residuals(X)
        values = cost(R)
        best = int(numpy.argmin(values))
        self.iteration += 1
        self.A, self.active = A, active
        if values[best] < self.value:
            self.damping *= self.trials[best] / 3
            self.x, self.r, self.value = X[best], R[best], float(values[best])
            self.history.append(self.value)
            return True
        self.damping *= 10 * self.trials[-1]
        self.history.append(self.value)
        return False


    def covariance(self):
        """
        (J^T J)^{-1} at the current point, on the free parameters; zero rows and columns for those
        held on a bound
        """
        J = self.problem.jacobian(self.x)
        A = J.T @ J
        active = free(self.x, J.T @ self.r, self.lower, self.upper)
        C = numpy.zeros_like(A)
        C[numpy.ix_(active, active)] = numpy.linalg.pinv(A[numpy.ix_(active, active)])
        return C


class LBFGS:
    """
    Projected L-BFGS: the two-loop recursion on the parameters not held on a bound, the step
    projected onto the bounds, and a backtracking line search whose step lengths are tried at once
    """


    def __init__(self, problem, x0, lower, upper, memory=10, lengths=(1, 0.5, 0.25, 0.1, 0.03, 0.01)):
        self.problem = problem
        self.lower, self.upper = numpy.asarray(lower, float), numpy.asarray(upper, float)
        self.x = numpy.clip(numpy.asarray(x0, float), self.lower, self.upper)
        self.memory, self.lengths = memory, numpy.asarray(lengths, float)
        self.r = problem.residuals(self.x[None, :])[0]
        self.value = float(cost(self.r[None, :])[0])
        self.g = problem.jacobian(self.x).T @ self.r
        self.pairs = []
        self.iteration = 0
        self.history = [self.value]
        return


    def direction(self, active):
        """
        -H g on the free parameters, by the two-loop recursion over the stored pairs
        """
        q = numpy.where(active, -self.g, 0.0)
        alphas = []
        for s, y in reversed(self.pairs):
            ρ = 1 / (y @ s)
            α = ρ * (s @ q)
            q -= α * y
            alphas.append((ρ, α))
        if self.pairs:
            s, y = self.pairs[-1]
            q *= (s @ y) / (y @ y)
        for (s, y), (ρ, α) in zip(self.pairs, reversed(alphas)):
            q += (α - ρ * (y @ q)) * s
        return numpy.where(active, q, 0.0)


    def step(self):
        """
        One iteration; False if no step length decreased the cost
        """
        active = free(self.x, self.g, self.lower, self.upper)
        d = self.direction(active)
        if d @ self.g >= 0:
            # not a descent direction: restart from steepest descent
            self.pairs = []
            d = numpy.where(active, -self.g, 0.0)
        # without curvature pairs, the longest step moves the parameters by at most one unit
        lengths = self.lengths if self.pairs else self.lengths / max(numpy.abs(d).max(), 1e-300)
        self.iteration += 1
        # the step lengths, shorter by 100 each time none of them decreases the cost
        for shrink in (1, 1e-2, 1e-4):
            X = numpy.clip(self.x + shrink * lengths[:, None] * d, self.lower, self.upper)
            R = self.problem.residuals(X)
            values = cost(R)
            armijo = values - self.value <= 1e-4 * ((X - self.x) @ self.g)
            best = int(numpy.argmax(armijo)) if armijo.any() else int(numpy.argmin(values))
            if values[best] < self.value:
                break
        else:
            self.pairs = []
            self.history.append(self.value)
            return False
        x, r = X[best], R[best]
        g = self.problem.jacobian(x).T @ r
        s, y = x - self.x, g - self.g
        if s @ y > 1e-12 * numpy.sqrt((s @ s) * (y @ y)):
            self.pairs = (self.pairs + [(s, y)])[-self.memory:]
        self.x, self.r, self.g, self.value = x, r, g, float(values[best])
        self.history.append(self.value)
        return True


    def projected_gradient(self):
        """
        The gradient on the free parameters, zero on those held on a bound
        """
        return numpy.where(free(self.x, self.g, self.lower, self.upper), self.g, 0.0)


# end of file
