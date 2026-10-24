#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Check that the Levenberg-Marquardt and the L-BFGS of the {optimize} action find the minimum of
the Rosenbrock function as least squares, and of a bounded, badly scaled linear least-squares
problem, where the projected gradient must vanish
"""


class Rosenbrock:
    """
    The Rosenbrock function as the residuals 10 (x[i+1] - x[i]^2) and 1 - x[i]
    """

    def residuals(self, X):
        import numpy
        X = numpy.atleast_2d(X)
        return numpy.hstack([10 * (X[:, 1:] - X[:, :-1] ** 2), 1 - X[:, :-1]])

    def jacobian(self, x):
        import numpy
        n = x.size
        J = numpy.zeros((2 * (n - 1), n))
        for i in range(n - 1):
            J[i, i], J[i, i + 1] = -20 * x[i], 10
            J[n - 1 + i, i] = -1
        return J


class Linear:
    """
    The residuals A x - b
    """

    def __init__(self, A, b):
        self.A, self.b = A, b

    def residuals(self, X):
        import numpy
        return numpy.atleast_2d(X) @ self.A.T - self.b

    def jacobian(self, x):
        return self.A


def minimize(method, problem, x0, lower, upper, iterations=2000):
    m = method(problem, x0, lower, upper)
    while m.iteration < iterations and m.step():
        pass
    return m


def test():
    import numpy
    from altar.actions.minimizers import LevenbergMarquardt, LBFGS, Scaled, free

    unbounded = numpy.full(10, numpy.inf)
    for method in (LevenbergMarquardt, LBFGS):
        m = minimize(method, Rosenbrock(), numpy.full(10, -1.0), -unbounded, unbounded)
        assert numpy.abs(m.x - 1).max() < 1e-6, (method.__name__, m.x)

    # columns over four decades, and a box that holds four of the six parameters on its faces
    rng = numpy.random.default_rng(3)
    A = rng.standard_normal((30, 6)) * numpy.array([1, 10, 100, 1, 0.1, 1])
    b = 5 * rng.standard_normal(30)
    lower, upper = numpy.full(6, -0.05), numpy.full(6, 0.05)
    for method in (LevenbergMarquardt, LBFGS):
        problem = Scaled(Linear(A, b), origin=numpy.zeros(6), scale=numpy.full(6, 0.025))
        m = minimize(method, problem, numpy.zeros(6), problem.to(lower), problem.to(upper))
        x = problem.back(m.x)
        g = A.T @ (A @ x - b)
        projected = numpy.where(free(x, g, lower, upper), g, 0.0)
        assert ((lower - 1e-12 <= x) & (x <= upper + 1e-12)).all()
        assert numpy.abs(projected).max() < 1e-4 * numpy.abs(g).max(), (method.__name__, projected)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
