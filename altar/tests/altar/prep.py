#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Check which parameter sets a model reports as not starting from their prior, the ones whose
evidence would be biased: none without a prep, or with the prior itself, or with a distribution of
the same kind and settings; those with other settings or another kind of distribution
"""


def test():
    import types
    import altar
    from altar.models.Bayesian import Bayesian

    prior = altar.distributions.uniform()(name="prior")
    prior.support = (-1.0, 20.0)
    same = altar.distributions.uniform()(name="same")
    same.support = (-1.0, 20.0)
    narrower = altar.distributions.uniform()(name="narrower")
    narrower.support = (-0.5, 20.0)
    gaussian = altar.distributions.gaussian()(name="gaussian")

    pset = lambda prep: types.SimpleNamespace(prior=prior, prep=prep)
    model = types.SimpleNamespace(psets={
        "none": pset(None), "itself": pset(prior), "same": pset(same),
        "narrower": pset(narrower), "gaussian": pset(gaussian),
    })
    assert Bayesian.unlike_prior(model) == ["narrower", "gaussian"]

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
