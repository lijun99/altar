#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Check that the controllers without annealing walk as many times as they should, counted from
where the run starts, a restart's iteration or 0: {mcmc}, {hmc} and {perturbation} their
{rounds}, {mala} once; and {perturbation} until an adaptive C_p settles
"""


def test():
    from altar.bayesian.controllers.Mcmc import Mcmc
    from altar.bayesian.controllers.Hmc import Hmc
    from altar.bayesian.controllers.Mala import Mala
    from altar.bayesian.controllers.Perturbation import Perturbation

    def walks(controller, start):
        controller.start_iteration = start
        iteration = start
        while controller.continuing(worker=None, iteration=iteration, tolerance=0):
            iteration += 1
        return iteration - start

    for factory in (Mcmc, Hmc, Perturbation):
        controller = factory(name=f"rounds-{factory.__name__}")
        controller.rounds = 3
        assert walks(controller, start=0) == 3, factory.__name__
        assert walks(controller, start=23) == 3, factory.__name__
    mala = Mala(name="once")
    assert walks(mala, start=0) == 1 and walks(mala, start=23) == 1

    # perturbation walks until an adaptive C_p settles: it changes 10x less every walk
    import types, numpy
    class Cp:
        latest = None
    perturbation = Perturbation(name="settle")
    perturbation.info = perturbation.warning = types.SimpleNamespace(log=lambda message: None)
    perturbation.model = types.SimpleNamespace(cp=Cp())
    perturbation.start_iteration = 0
    iteration = 0
    while perturbation.continuing(worker=None, iteration=iteration, tolerance=0):
        iteration += 1
        perturbation.model.cp.latest = numpy.identity(2) * (1 + 10.0 ** -iteration)
    # changes of 0.09, then 0.009 from walk 2 to 3, below 0.01
    assert iteration == 3, iteration
    perturbation.max_rounds = 2
    perturbation.model.cp.latest, perturbation._cp, iteration = None, None, 0
    while perturbation.continuing(worker=None, iteration=iteration, tolerance=0):
        iteration += 1
        perturbation.model.cp.latest = numpy.identity(2) * (1 + 10.0 ** -iteration)
    assert iteration == 2, iteration

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
