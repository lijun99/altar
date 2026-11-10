#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
The 9patch static inversion with bounds that bind: the dip slips uniform on (low, high), low
near their posterior means, sampled by the perturbation controller and by CATMIP, against the
exact posterior from rejection sampling of the untruncated gaussian, and the exact evidence

    python bounded.py                  # dip slips on (0.95, 20)
    python bounded.py --low=0.9 --catmip
"""

import argparse
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile
import h5py
import numpy

HERE = pathlib.Path(__file__).resolve().parent
EXAMPLES = HERE.parent.parent / "examples"
PATCHES, SIGMA = 9, 0.5
# the tolerances, for 1024 chains: the mean within this many standard deviations, the standard
# deviations within this ratio, the correlations within this difference, and the evidence within
# this many nats, exact draws and GHK for perturbation, an annealing estimate for catmip
MEAN, SD, CORRELATION = 0.2, (0.85, 1.15), 0.2
EVIDENCE = {"perturbation": 0.3, "catmip": 3.0}


def exact(low, high, draws):
    """
    The mean, the covariance and the log mass of the box of the posterior, from the draws of the
    untruncated one that land in the box, and the log evidence
    """
    folder = EXAMPLES / "9patch"
    G = numpy.asarray(h5py.File(folder / "static.gf.h5")["static.gf"], dtype=float)
    d = numpy.asarray(h5py.File(folder / "static.data.h5")["static.data"], dtype=float)
    C = numpy.asarray(h5py.File(folder / "static.Cd.h5")["static.Cd"], dtype=float)
    observations, n = G.shape
    W = numpy.linalg.inv(C)
    # the strike slips gaussian, the dip slips flat
    P = numpy.diag([1 / SIGMA ** 2] * PATCHES + [0] * PATCHES)
    A = G.T @ W @ G + P
    S = numpy.linalg.inv(A)
    S = (S + S.T) / 2
    m = S @ (G.T @ W @ d)
    rng = numpy.random.default_rng(7)
    kept, total = [], 0
    for _ in range(0, draws, 1_000_000):
        x = rng.multivariate_normal(m, S, size=1_000_000)
        dip = x[:, PATCHES:]
        kept.append(x[((dip > low) & (dip < high)).all(axis=1)])
        total += x.shape[0]
    x = numpy.vstack(kept)
    log_mass = math.log(x.shape[0] / total)
    # log ∫ N(d; Gθ, C) N(θ_s; 0, σ²) dθ, the dip slips with a density of 1, then the box
    flat = (-0.5 * (d @ W @ d - m @ A @ m) + 0.5 * (n - observations - PATCHES) * math.log(2 * math.pi)
            - 0.5 * numpy.linalg.slogdet(A)[1] - 0.5 * numpy.linalg.slogdet(C)[1]
            - PATCHES * math.log(SIGMA))
    evidence = flat + log_mass - PATCHES * math.log(high - low)
    return x.mean(axis=0), numpy.cov(x, rowvar=False), log_mass, x.shape[0], evidence


def run(name, settings, low, high):
    """
    The physical samples of the final step of a run of the static example, and its evidence
    """
    scratch = pathlib.Path(tempfile.mkdtemp(prefix=f"altar-bounded-{name}-"))
    try:
        shutil.copytree(EXAMPLES / "9patch", scratch / "9patch")
        shutil.copy(EXAMPLES / "static_perturbation.pfg", scratch)
        command = ["slipmodel.plexus", "sample", "--config=static_perturbation.pfg", "--job.gpus=0",
                   f"--model.psets.dipslip.prior.support=({low},{high})",
                   f"--model.psets.dipslip.prep=uniform",
                   f"--model.psets.dipslip.prep.support=({low},{high})",
                   "--controller.archiver.output_dir=results", *settings]
        status = subprocess.run(command, cwd=scratch, capture_output=True, text=True)
        if status.returncode != 0:
            print(status.stdout[-3000:], status.stderr[-3000:])
            raise SystemExit(f"{name} failed")
        for line in status.stdout.splitlines():
            if "perturbation:" in line and ("bounces" in line or "evidence" in line):
                print(f"  {line.strip()}")
        with h5py.File(scratch / "results" / "step_final.h5") as h5:
            sets = h5["ParameterSets"]
            pick = lambda name: sets[f"{name}_physical" if f"{name}_physical" in sets else f"{name}_sampling"]
            theta = numpy.hstack([numpy.asarray(pick("strikeslip")), numpy.asarray(pick("dipslip"))])
            evidence = float(numpy.asarray(h5["Annealer/log_evidence"]))
        return theta, evidence
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--low", type=float, default=0.95)
    parser.add_argument("--high", type=float, default=20.0)
    parser.add_argument("--draws", type=int, default=10_000_000)
    parser.add_argument("--steps", type=int, default=20, help="the trajectories of the perturbation run")
    parser.add_argument("--catmip", action="store_true", help="also run catmip")
    options = parser.parse_args()
    low, high = options.low, options.high

    mean, cov, log_mass, kept, evidence = exact(low, high, options.draws)
    sd = numpy.sqrt(numpy.diag(cov))
    print(f"exact: dip slips on ({low}, {high}), box mass exp({log_mass:.4f}) from {kept} of "
          f"{options.draws} draws, log evidence {evidence:.3f}")
    runs = {"perturbation": [f"--job.steps={options.steps}"]}
    if options.catmip:
        runs["catmip"] = ["--controller=altar.bayesian.catmip", "--job.steps=100"]
    failures = 0
    for name, settings in runs.items():
        theta, estimate = run(name, settings, low, high)
        shift = numpy.abs(theta.mean(axis=0) - mean) / sd
        ratio = theta.std(axis=0) / sd
        drift = numpy.abs(numpy.corrcoef(theta, rowvar=False) - cov / numpy.outer(sd, sd)).max()
        good = (shift.max() <= MEAN and SD[0] <= ratio.min() and ratio.max() <= SD[1]
                and drift <= CORRELATION and abs(estimate - evidence) <= EVIDENCE[name])
        failures += not good
        print(f"{name:12s} {'ok' if good else 'FAIL':4s} mean {shift.max():.3f} sd, sd ratio "
              f"[{ratio.min():.3f}, {ratio.max():.3f}], correlation {drift:.3f}, "
              f"log evidence {estimate:.3f} (exact {evidence:.3f})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


# end of file
