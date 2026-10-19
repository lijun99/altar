#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Check the resampling methods: each keeps n samples, copies a sample of weight w n w times on
average, and keeps to its own bounds: systematic to the floor or the ceiling of n w, residual to
at least its floor; the counts of multinomial stray the most; and that the resampling of {COV}
moves every part of a reparameterized step together
"""


def test():
    import journal
    import numpy
    from altar.bayesian.statistics import multiplicities, RESAMPLING
    from altar.bayesian.states.CoolingStep import CoolingStep
    from altar.bayesian.schedulers.COV import COV
    from altar.bayesian.resamplers.ImportanceResampler import ImportanceResampler

    n, trials = 200, 4000
    rng = numpy.random.default_rng(0)
    # lognormal weights, as from a β step at a coefficient of variation of about 1
    w = numpy.exp(rng.normal(scale=numpy.sqrt(numpy.log(2)), size=n))
    w /= w.sum()
    spread = {}
    for method in RESAMPLING:
        counts = numpy.array([multiplicities(w=w, rng=rng, method=method) for _ in range(trials)])
        assert (counts.sum(axis=1) == n).all(), method
        # unbiased, within five standard errors of the mean count of each sample
        error = numpy.sqrt(counts.var(axis=0) / trials) + 1e-12
        assert (numpy.abs(counts.mean(axis=0) - n * w) <= 5 * error + 1e-9).all(), method
        spread[method] = counts.var(axis=0).sum()
    systematic = numpy.array([multiplicities(w=w, rng=rng, method="systematic") for _ in range(100)])
    assert ((systematic == numpy.floor(n * w)) | (systematic == numpy.ceil(n * w))).all()
    residual = numpy.array([multiplicities(w=w, rng=rng, method="residual") for _ in range(100)])
    assert (residual >= numpy.floor(n * w)).all()
    assert spread["multinomial"] == max(spread.values())
    assert spread["systematic"] < 0.25 * spread["multinomial"]

    # the resampling of COV, through its resampler: tag every part of the state of sample i
    step = CoolingStep.alloc(samples=n, parameters=2, has_reparametrization=True)
    tags = numpy.arange(n, dtype=float)
    step.theta_sampling[...] = tags[:, None]
    step.theta[...] = -tags[:, None]
    step.jacobian[...] = 2 * tags
    step.prior[...] = 3 * tags
    step.data[...] = 4 * tags
    step.posterior[...] = 5 * tags
    resampler = ImportanceResampler(name="resampling-test")
    resampler.rng, resampler.info = numpy.random.default_rng(1), journal.info("resampling")
    rows = resampler.rows(w=w)
    step.reorder(rows=rows)
    origin = step.theta_sampling[:, 0].astype(int)
    assert numpy.array_equal(origin, rows)
    for part, factor in ((step.theta[:, 0], -1), (step.jacobian, 2), (step.prior, 3),
                         (step.data, 4), (step.posterior, 5)):
        assert numpy.array_equal(part, factor * origin)
    # the default method of the component, and of the scheduler's resampler
    assert ImportanceResampler(name="default-test").method == "systematic"
    assert isinstance(COV(name="cov-test").resampler, ImportanceResampler)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
