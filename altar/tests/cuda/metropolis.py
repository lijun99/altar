#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved


"""
Sanity check: altar.cuda.libcudaaltar.metropolis's bindings, against a real GPU.

Covers the full Metropolis-Hastings step in order: {setValidSampleIndices} (compacts the
not-invalid indices to the front, and the resulting count lands in a device scalar the host
reads back), {queueValidSamples} (gathers the corresponding rows of a proposal matrix into a
candidate matrix), and {metropolisUpdate} (the actual accept/reject test -- checked against
the exact analytic rule log(dice) <= posterior_candidate - posterior[sample_index], and that
only accepted rows get overwritten).
"""


def test():
    import numpy
    import pyre.grid
    import altar
    import altar.cuda

    metropolis = altar.cuda.libcudaaltar.metropolis

    samples = 10
    parameters = 3

    invalid = pyre.grid.managed(shape=(samples,), cell="int32")
    numpy.asarray(invalid)[:] = 0
    numpy.asarray(invalid)[[2, 5, 7]] = 1

    valid_indices = pyre.grid.managed(shape=(samples,), cell="int32")
    valid_count = pyre.grid.managed(shape=(1,), cell="int32")
    metropolis.cudaMetropolis_setValidSampleIndices(valid_indices, invalid, valid_count)

    count = int(numpy.asarray(valid_count)[0])
    assert count == samples - 3
    vi = numpy.asarray(valid_indices)[:count]
    assert sorted(vi.tolist()) == sorted(set(range(samples)) - {2, 5, 7})

    # queueValidSamples: gather the valid rows of a proposal matrix
    theta_proposal = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta_proposal)[:, :] = numpy.arange(samples * parameters).reshape(samples, parameters)
    theta_candidate = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta_candidate)[:, :] = -1.0
    metropolis.cudaMetropolis_queueValidSamples(theta_candidate, theta_proposal, valid_indices, count)

    tc = numpy.asarray(theta_candidate)
    tp = numpy.asarray(theta_proposal)
    for s in range(count):
        assert numpy.array_equal(tc[s], tp[vi[s]])

    # metropolisUpdate: half the candidates should beat the current posterior, half shouldn't
    rng = numpy.random.default_rng(1)
    batch = count

    theta = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta)[:, :] = 0.0
    prior = pyre.grid.managed(shape=(samples,), cell="float64")
    numpy.asarray(prior)[:] = 0.0
    data = pyre.grid.managed(shape=(samples,), cell="float64")
    numpy.asarray(data)[:] = 0.0
    posterior = pyre.grid.managed(shape=(samples,), cell="float64")
    numpy.asarray(posterior)[:] = -10.0

    prior_candidate = pyre.grid.managed(shape=(batch,), cell="float64")
    numpy.asarray(prior_candidate)[:] = 1.0
    data_candidate = pyre.grid.managed(shape=(batch,), cell="float64")
    numpy.asarray(data_candidate)[:] = 2.0

    posterior_values = rng.uniform(-20, -5, size=batch)
    posterior_candidate = pyre.grid.managed(shape=(batch,), cell="float64")
    numpy.asarray(posterior_candidate)[:] = posterior_values

    dice_values = rng.uniform(0.01, 0.99, size=batch)
    dices = pyre.grid.managed(shape=(batch,), cell="float64")
    numpy.asarray(dices)[:] = dice_values

    acceptance_flag = pyre.grid.managed(shape=(batch,), cell="int32")
    numpy.asarray(acceptance_flag)[:] = 0

    metropolis.cudaMetropolis_metropolisUpdate(
        theta, prior, data, posterior,
        theta_candidate, prior_candidate, data_candidate, posterior_candidate,
        dices, acceptance_flag, valid_indices, batch,
    )

    expected_accept = numpy.log(dice_values) <= (posterior_values - (-10.0))
    assert numpy.array_equal(numpy.asarray(acceptance_flag).astype(bool), expected_accept)

    th, pr = numpy.asarray(theta), numpy.asarray(prior)
    da, po = numpy.asarray(data), numpy.asarray(posterior)
    for s in range(batch):
        idx = vi[s]
        if expected_accept[s]:
            assert numpy.array_equal(th[idx], tc[s])
            assert pr[idx] == 1.0 and da[idx] == 2.0 and po[idx] == posterior_values[s]
        else:
            assert numpy.array_equal(th[idx], numpy.zeros(parameters))
            assert pr[idx] == 0.0 and po[idx] == -10.0

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
