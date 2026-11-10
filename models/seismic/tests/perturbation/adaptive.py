#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
The 9patchCp static inversion with an adaptive C_p, re-estimated from the mean model at every
walk: the perturbation controller over several rounds, and CATMIP, against the fixed point, the
mean m whose posterior under C_d + C_p(m) has mean m, and the gaussian posterior there

    python adaptive.py                 # perturbation, until C_p settles
    python adaptive.py --rounds=6 --catmip
    python adaptive.py --gpu
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import h5py
import numpy

HERE = pathlib.Path(__file__).resolve().parent
EXAMPLES = HERE.parent.parent / "examples"
CASE = EXAMPLES / "9patchCp"
PATCHES, SIGMA = 9, 0.7
# the tolerances, for 1024 chains, as in bounded.py
MEAN, SD, CORRELATION = 0.2, (0.85, 1.15), 0.2


def load(name):
    with h5py.File(CASE / name) as h5:
        return {key: numpy.asarray(h5[key], dtype=float) for key in h5}


def posterior(cp):
    """
    The mean and the covariance of the posterior under C_d + {cp}, the dip slips flat
    """
    G, d, Cd = load("static.gf.h5")["static.gf"], load("static.data.h5")["static.data"], load("static.Cd.h5")["static.Cd"]
    W = numpy.linalg.inv(Cd + cp)
    A = G.T @ W @ G + numpy.diag([1 / SIGMA ** 2] * PATCHES + [0] * PATCHES)
    S = numpy.linalg.inv(A)
    S = (S + S.T) / 2
    return S @ (G.T @ W @ d), S


def cp(theta):
    """
    C_p = K_p C_mu K_p^T, K_p[:, i] = K_i theta
    """
    cmu = numpy.atleast_2d(load("static.Cmu.h5")["static.Cmu"])
    kernels = load("static.kernel.h5")
    order = lambda key: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", key)]
    Kp = numpy.column_stack([kernels[key] @ theta for key in sorted(kernels, key=order)])
    return Kp @ cmu @ Kp.T


def fixed_point(start, iterations=50):
    """
    Iterate m -> the posterior mean under C_p(m) from {start}; the trace of C_p at every step
    """
    m, traces = start, []
    for _ in range(iterations):
        c = cp(m)
        traces.append(numpy.trace(c))
        new, S = posterior(c)
        if numpy.abs(new - m).max() < 1e-12:
            break
        m = new
    return m, S, traces


def run(name, settings, gpus=0, seed=0):
    """
    The samples of the final step of a run of staticCp.pfg, and its log
    """
    scratch = pathlib.Path(tempfile.mkdtemp(prefix=f"altar-adaptive-{name}-"))
    try:
        shutil.copytree(CASE, scratch / "9patchCp")
        shutil.copy(EXAMPLES / "staticCp.pfg", scratch)
        command = ["slipmodel.plexus", "sample", "--config=staticCp.pfg", f"--job.gpus={gpus}",
                   "--controller.archiver.output_dir=results", f"--rng.seed={seed}", *settings]
        status = subprocess.run(command, cwd=scratch, capture_output=True, text=True)
        if status.returncode != 0:
            print(status.stdout[-3000:], status.stderr[-3000:])
            raise SystemExit(f"{name} failed")
        with h5py.File(scratch / "results" / "step_final.h5") as h5:
            sets = h5["ParameterSets"]
            pick = lambda s: sets[f"{s}_physical" if f"{s}_physical" in sets else f"{s}_sampling"]
            theta = numpy.hstack([numpy.asarray(pick("strikeslip")), numpy.asarray(pick("dipslip"))])
        return theta, status.stdout
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rounds", type=int, default=None, help="the least walks of perturbation")
    parser.add_argument("--catmip", action="store_true", help="also run catmip")
    parser.add_argument("--gpu", action="store_true", help="run on the gpu")
    parser.add_argument("--seed", type=int, default=0, help="the seed of the runs")
    options = parser.parse_args()

    initial = load("static.initialModel.h5")["static.initialModel"]
    m, S, traces = fixed_point(start=initial)
    sd = numpy.sqrt(numpy.diag(S))
    print(f"fixed point: {len(traces)} iterations from the initial model, trace C_p "
          + " ".join(f"{t:.4g}" for t in traces[:6]) + f" ... {traces[-1]:.6g}")
    runs = {"perturbation": ["--controller=altar.bayesian.perturbation"]
                            + ([f"--controller.rounds={options.rounds}"] if options.rounds else [])}
    if options.catmip:
        runs["catmip"] = []
    failures = 0
    for name, settings in runs.items():
        theta, log = run(name, settings, gpus=int(options.gpu), seed=options.seed)
        for line in log.splitlines():
            if "C_p" in line or "perturbation: log evidence" in line:
                print(f"  {line.strip()}")
        shift = numpy.abs(theta.mean(axis=0) - m) / sd
        ratio = theta.std(axis=0) / sd
        drift = numpy.abs(numpy.corrcoef(theta, rowvar=False) - S / numpy.outer(sd, sd)).max()
        good = shift.max() <= MEAN and SD[0] <= ratio.min() and ratio.max() <= SD[1] and drift <= CORRELATION
        failures += not good
        print(f"{name:12s} {'ok' if good else 'FAIL':4s} mean {shift.max():.3f} sd, sd ratio "
              f"[{ratio.min():.3f}, {ratio.max():.3f}], correlation {drift:.3f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


# end of file
