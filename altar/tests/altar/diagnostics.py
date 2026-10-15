#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Check the diagnostics of the walks on a pooled population over two tasks: the rows of the
chains, the parents and lineages through a resampling and a walk, the correlation of a walk that
doesn't move and of one that forgets its start, and the effective states per chain of a pool of
independent states and of one of copies; and that a pool knows the slot of the final states
"""


def test():
    import numpy
    from altar.bayesian.methods.Diagnostics import Diagnostics
    from altar.bayesian.methods.Pool import Pool

    tasks, chains, pool, parameters = 2, 2000, 4, 3
    rows = tasks * chains * pool
    rng = numpy.random.default_rng(0)

    # task 1's chains in slot 2 follow task 0's block of 4 slots
    diagnostics = Diagnostics(tasks=tasks, chains=chains, pool=pool)
    slot2 = diagnostics.rows(slot=2)
    assert slot2.size == tasks * chains
    assert slot2[0] == 2 * chains and slot2[chains] == chains * pool + 2 * chains

    # resample to the first half of the rows, each twice
    resampled = numpy.repeat(numpy.arange(rows // 2), 2)
    rng.shuffle(resampled)
    diagnostics.resampled(rows=resampled)
    assert diagnostics.parents == rows // 2
    # walks that don't move: the final states are the starting ones
    θ = rng.normal(size=(rows, parameters))
    diagnostics.begin(θ=θ)
    still = θ.copy()
    for slot in range(pool):
        still[diagnostics.rows(slot=slot)] = θ[diagnostics.rows(slot=0)]
    walk = diagnostics.end(θ=still, final=3, iteration=1, beta=0.5)
    assert abs(walk["correlation"] - 1) < 1e-12
    # copies of one state are one effective state
    assert abs(walk["states"] - 1) < 1e-9
    # every row descends from its chain's starting row
    lineage = diagnostics.lineage.reshape(tasks, pool, chains)
    assert (lineage == lineage[:, :1, :]).all()
    starts = resampled[diagnostics.rows(slot=0)]
    assert walk["lineages"] == numpy.unique(starts).size

    # walks that forget their start, keeping independent states
    diagnostics.begin(θ=still)
    fresh = rng.normal(size=(rows, parameters))
    walk = diagnostics.end(θ=fresh, final=1, iteration=2, beta=1.0)
    assert abs(walk["correlation"]) < 0.05 and walk["correlation_max"] < 0.1
    assert abs(walk["states"] - pool) < 0.5
    # no resampling: every row a parent
    assert walk["parents"] == rows
    assert len(diagnostics.history) == 2

    # a pool of 3 slots, every 2 steps, over a walk of 7: kept at 2, 4, 6 and the final 7
    kept = []
    p = Pool(size=3, interval=2, chains=10)
    p.begin()
    for _ in range(7):
        p.advance(kept.append)
    p.end(kept.append)
    assert kept == [0, 10, 20, 0] and p.final == 0
    # the next walk of 2 keeps at 2, its final state too, in slot 1, and fills slots 2 and 0
    kept = []
    p.begin()
    for _ in range(2):
        p.advance(kept.append)
    p.end(kept.append)
    assert kept == [10, 20, 0] and p.final == 1

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
