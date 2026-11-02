#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Check that the CMA-ES of the {optimize} action finds the minima of an ill-conditioned ellipsoid
and of the Rosenbrock function, and a minimum outside its bounds on the bounds
"""


def minimize(f, x0, scale, lower=None, upper=None, generations=3000):
    from altar.actions.cmaes import CMAES

    es = CMAES(x0=x0, scale=scale, lower=lower, upper=upper, seed=1)
    while es.generation < generations and not es.stop(tolx=1e-10, tolfun=1e-14):
        x, penalty = es.ask()
        es.tell(x, [f(v) for v in x], penalty)
    return es


def test():
    import numpy

    # an ellipsoid with axes over three decades
    def ellipsoid(x):
        return sum(10 ** (6 * i / (len(x) - 1)) * v * v for i, v in enumerate(x))
    es = minimize(ellipsoid, x0=3 * numpy.ones(10), scale=2.0)
    assert es.best[0] < 1e-12

    # the curved valley of the Rosenbrock function, with its minimum at 1
    def rosenbrock(x):
        return sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    es = minimize(rosenbrock, x0=numpy.zeros(10), scale=0.5)
    assert numpy.abs(es.best[1] - 1).max() < 1e-6

    # a sphere centered partly outside the box: the minimum is on its faces
    center = numpy.array([3, -3, 0.5, 2, -1])
    es = minimize(lambda x: ((x - center) ** 2).sum(), x0=numpy.zeros(5), scale=1.0,
                  lower=-numpy.ones(5), upper=numpy.ones(5))
    assert numpy.allclose(es.best[1], numpy.clip(center, -1, 1), atol=1e-6)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
