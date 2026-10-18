#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Check that the controllers without annealing walk as many times as they should, counted from
where the run starts, a restart's iteration or 0: {mcmc} and {hmc} their {rounds}, {mala} once
"""


def test():
    from altar.bayesian.controllers.Mcmc import Mcmc
    from altar.bayesian.controllers.Hmc import Hmc
    from altar.bayesian.controllers.Mala import Mala

    def walks(controller, start):
        controller.start_iteration = start
        iteration = start
        while controller.continuing(worker=None, iteration=iteration, tolerance=0):
            iteration += 1
        return iteration - start

    for factory in (Mcmc, Hmc):
        controller = factory(name=f"rounds-{factory.__name__}")
        controller.rounds = 3
        assert walks(controller, start=0) == 3, factory.__name__
        assert walks(controller, start=23) == 3, factory.__name__
    mala = Mala(name="once")
    assert walks(mala, start=0) == 1 and walks(mala, start=23) == 1

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
