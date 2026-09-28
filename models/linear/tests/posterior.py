#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Check the samplers against the exact posterior of the linear example, patch-9: its data are
linear in the parameters with gaussian noise, so under a gaussian prior the posterior is the
gaussian N(μ, A⁻¹), A = Gᵀ C_d⁻¹ G + I/σ², μ = A⁻¹ Gᵀ C_d⁻¹ d. The cases with a wide uniform
prior, reparameterized, compare against the same posterior without the prior term, the
truncation being negligible there.

Each case runs {altar-linear} in a scratch directory and compares the mean, the standard
deviations and the correlations of its final samples with the exact ones.

    python posterior.py                 # every case, on the cpu
    python posterior.py mala hmc        # some of them
    python posterior.py --gpu           # on the gpu
    python posterior.py --list          # the cases
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import h5py
import numpy


# the example, relative to me
EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
CASE = "patch-9"
# the prior of the gaussian cases, and the support of the uniform ones
SIGMA = 0.5
SUPPORT = (-5, 5)

# the configuration every case starts from
BASE = f"""
linear:
    model = altar.models.linear
    model:
        case = {CASE}
        parameters = 18
        dataobs:
            observations = 108
            cd_file = cd.txt
        psets_list = [all]
        psets:
            all = contiguous
            all:
                count = {{linear.model.parameters}}
                prep = gaussian
                prep.sigma = {SIGMA}
                prior = gaussian
                prior.sigma = {SIGMA}
    controller:
        archiver = altar.bayesian.h5recorder
        archiver.output_dir = results
    job.tasks = 1
    job.chains = 2**8
"""

UNIFORM = [
    "--model.psets.all.prior=uniform",
    f"--model.psets.all.prior.support=({SUPPORT[0]},{SUPPORT[1]})",
    "--model.psets.all.prior.reparameterize=True",
]

# name: (the controller and sampler settings, whether the prior is the wide uniform one)
CASES = {
    "catmip": (["--controller=altar.bayesian.catmip", "--job.steps=256"], False),
    "mcmc": (["--controller=altar.bayesian.mcmc", "--controller.rounds=16", "--job.steps=256"], False),
    "catmip_hmc": (["--controller=altar.bayesian.catmip_hmc", "--job.steps=20"], False),
    "hmc": (["--controller=altar.bayesian.hmc", "--job.steps=200"], False),
    "catmip_mala": (["--controller=altar.bayesian.catmip_mala", "--job.steps=200"], False),
    "mala": (["--controller=altar.bayesian.mala", "--job.steps=4000"], False),
    "catmip-uniform": (["--controller=altar.bayesian.catmip", "--job.steps=256"], True),
    "hmc-uniform": (["--controller=altar.bayesian.hmc", "--job.steps=200"], True),
    "mala-uniform": (["--controller=altar.bayesian.mala", "--job.steps=4000"], True),
}

# the tolerances, for 256 chains: the mean within this many posterior standard deviations,
# the standard deviations within this ratio, the correlations within this difference; 256
# independent exact draws reach 0.23, [0.83, 1.19] and 0.25 at worst over 200 trials
MEAN = 0.3
SD = (0.8, 1.25)
CORRELATION = 0.3


def exact(prior):
    """
    The mean and the covariance of the posterior, with the gaussian prior or without a prior
    """
    folder = EXAMPLES / CASE
    G = numpy.loadtxt(folder / "green.txt")
    d = numpy.loadtxt(folder / "data.txt")
    Cd = numpy.loadtxt(folder / "cd.txt")
    W = numpy.linalg.inv(Cd)
    A = G.T @ W @ G
    if prior:
        A += numpy.eye(A.shape[0]) / SIGMA**2
    cov = numpy.linalg.inv(A)
    return cov @ (G.T @ W @ d), cov


def samples(results):
    """
    The physical samples of the final step
    """
    parameters = h5py.File(results / "step_final.h5")["ParameterSets"]
    name = "all_physical" if "all_physical" in parameters else "all_sampling"
    return numpy.asarray(parameters[name])


def compare(theta, mean, cov):
    """
    How far the samples are from the exact posterior
    """
    sd = numpy.sqrt(numpy.diag(cov))
    correlation = cov / numpy.outer(sd, sd)
    shift = numpy.abs(theta.mean(axis=0) - mean) / sd
    ratio = theta.std(axis=0) / sd
    drift = numpy.abs(numpy.corrcoef(theta, rowvar=False) - correlation).max()
    good = shift.max() <= MEAN and SD[0] <= ratio.min() and ratio.max() <= SD[1] and drift <= CORRELATION
    return good, shift.max(), ratio.min(), ratio.max(), drift


def run(name, gpu, precision, keep):
    """
    Run one case and compare it with the exact posterior
    """
    settings, uniform = CASES[name]
    scratch = pathlib.Path(tempfile.mkdtemp(prefix=f"altar-posterior-{name}-"))
    try:
        shutil.copytree(EXAMPLES / CASE, scratch / CASE)
        (scratch / "posterior.pfg").write_text(BASE)
        command = ["altar-linear", "--config=posterior.pfg", f"--job.gpus={int(gpu)}", *settings]
        if gpu:
            command.append(f"--job.gpuprecision={precision}")
        if uniform:
            command += UNIFORM
        start = time.perf_counter()
        status = subprocess.run(command, cwd=scratch, capture_output=True, text=True, errors="replace")
        elapsed = time.perf_counter() - start
        if status.returncode != 0:
            (scratch / "run.log").write_text(status.stdout + status.stderr)
            keep = True
            return False, f"{name:18s} FAILED to run ({status.returncode}); see {scratch}/run.log"
        mean, cov = exact(prior=not uniform)
        good, shift, low, high, drift = compare(samples(scratch / "results"), mean, cov)
        verdict = "ok" if good else "FAIL"
        return good, (f"{name:18s} {verdict:4s} {elapsed:6.0f}s  mean {shift:.2f} sd  "
                      f"sd ratio [{low:.2f}, {high:.2f}]  correlation {drift:.2f}")
    finally:
        if keep:
            print(f"  kept {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", nargs="*", help="the cases to run; all by default")
    parser.add_argument("--gpu", action="store_true", help="run on the gpu")
    parser.add_argument("--precision", default="float64", help="the gpu precision")
    parser.add_argument("--keep", action="store_true", help="keep the scratch directories")
    parser.add_argument("--list", action="store_true", help="list the cases")
    options = parser.parse_args()

    if options.list:
        for name, (settings, uniform) in CASES.items():
            print(f"{name:18s} {' '.join(settings)}{' (wide uniform prior)' if uniform else ''}")
        return 0
    if shutil.which("altar-linear") is None:
        print("altar-linear is not on the PATH")
        return 1
    unknown = [name for name in options.cases if name not in CASES]
    if unknown:
        print(f"unknown case(s): {', '.join(unknown)}; see --list")
        return 1

    print(f"tolerances: mean within {MEAN} sd, sd ratio in [{SD[0]}, {SD[1]}], "
          f"correlation within {CORRELATION}")
    failures = 0
    for name in options.cases or CASES:
        good, line = run(name, gpu=options.gpu, precision=options.precision, keep=options.keep)
        print(line, flush=True)
        failures += not good
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


# end of file
