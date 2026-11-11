#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# compare the samples of a run on the synthetic case with the exact gaussian posterior (the
# priors of trajectory.pfg): per group, the median distance of the sampled means from the exact
# ones in exact standard deviations, and the median ratio of the sampled spreads to the exact
# ones; also the range of the eigenvalues of each block's posterior precision, which sets the
# step sizes and the mixing of the gradient-based samplers
#     python compare.py results_sgld [results_hmc ...]
import sys
from pathlib import Path

import h5py
import numpy

from altar.models.gnss import design

SIGMA = {"secular": 10.0, "seasonal": 3.0, "transient": 1.0, "steps": 10.0}


def snapshot(path):
    """
    The final snapshot of a run's results directory, or its last numbered one (the cuda sgld
    method writes only those)
    """
    path = Path(path)
    final = path / "step_final.h5"
    return final if final.exists() else sorted(path.glob("step_*.h5"))[-1]


def samples(path, name, columns):
    """
    The samples of the parameter set {name}: its own dataset, or its {columns} of the whole
    "theta" (the cuda sgld snapshots)
    """
    with h5py.File(snapshot(path), "r") as h5:
        ps = h5["ParameterSets"]
        for key in (f"{name}_physical", name, f"{name}_sampling"):
            if key in ps:
                return ps[key][:]
        if "theta" in ps:
            return ps["theta"][:, columns]
    raise KeyError(name)


def main(runs):
    d = design.load("trajectory.h5")
    offsets, o = {}, 0
    for g in d.groups:
        offsets[g.name], o = o, o + d.blocks * g.width
    mu, sd, covariances = d.posterior(offsets, SIGMA)
    eig = numpy.concatenate([1 / numpy.linalg.eigvalsh(C) for C in covariances])
    print(f"posterior precision eigenvalues {eig.min():.3g} .. {eig.max():.3g} "
          f"(condition {eig.max() / eig.min():.3g})")
    truth = numpy.load("truth.npy")
    z = (truth - mu) / sd
    print(f"truth vs exact posterior: median |z| {numpy.median(numpy.abs(z)):.2f}, "
          f"95% within 2 sd: {(numpy.abs(z) < 2).mean():.2f}")
    for run in runs:
        print(f"== {run} ({snapshot(run).name})")
        for g in d.groups:
            sl = slice(offsets[g.name], offsets[g.name] + d.blocks * g.width)
            x = samples(run, g.name, sl)
            zm = (x.mean(axis=0) - mu[sl]) / sd[sl]
            ratio = x.std(axis=0) / sd[sl]
            print(f"  {g.name:10s} {x.shape[0]} samples: |mean - exact| / sd median "
                  f"{numpy.median(numpy.abs(zm)):.2f} max {numpy.abs(zm).max():.2f}; "
                  f"sd / exact sd median {numpy.median(ratio):.2f} "
                  f"[{numpy.percentile(ratio, 5):.2f}, {numpy.percentile(ratio, 95):.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


# end of file
