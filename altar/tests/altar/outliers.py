#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Check that the burn-in of {ConstantTemperature} replaces the outlier chains, and only them, by
copies of other chains, with every part of their state
"""


def test():
    import journal
    import numpy
    from altar.bayesian.states.CoolingStep import CoolingStep
    from altar.bayesian.schedulers.ConstantTemperature import ConstantTemperature

    samples, parameters = 1000, 3
    rng = numpy.random.default_rng(0)
    step = CoolingStep.alloc(samples=samples, parameters=parameters, has_reparametrization=True)
    # tag every part of the state of chain i with i, so the rows can be traced
    tags = numpy.arange(samples, dtype=float)
    step.theta_sampling[...] = tags[:, None]
    step.theta[...] = -tags[:, None]
    step.jacobian[...] = 2 * tags
    step.prior[...] = 3 * tags
    # regular chains with no tail below the cutoff
    step.data[...] = rng.uniform(-1, 1, size=samples)
    # a few chains stuck far below the others
    stuck = numpy.array([5, 17, 400])
    step.data[stuck] = -1e3
    step.posterior[...] = step.data
    before = step.data.copy()

    scheduler = ConstantTemperature(name="outliers")
    scheduler.rng = numpy.random.default_rng(1)
    scheduler.info = journal.info("outliers")
    scheduler.burnin = 1
    scheduler.walks = 1
    scheduler.replace_outliers(step=step)

    # where each chain came from, read off its tag
    origin = step.theta_sampling[:, 0].astype(int)
    replaced = numpy.flatnonzero(origin != tags)
    assert set(replaced) == set(stuck)
    assert not set(origin[stuck]) & set(stuck)
    # every part of the state moved together
    assert numpy.array_equal(step.theta[:, 0], -origin)
    assert numpy.array_equal(step.jacobian, 2 * origin)
    assert numpy.array_equal(step.prior, 3 * origin)
    assert numpy.array_equal(step.data, before[origin])
    assert numpy.array_equal(step.posterior, before[origin])

    # nothing to replace the second time
    again = step.theta_sampling.copy()
    scheduler.replace_outliers(step=step)
    assert numpy.array_equal(step.theta_sampling, again)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
