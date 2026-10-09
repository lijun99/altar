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
truncation being negligible there; the cases with a tight uniform prior, which cuts into that
posterior, compare against its moments within the support, from rejection sampling.

Each case runs {altar-linear} in a scratch directory and compares the mean, the standard
deviations and the correlations of its final samples with the exact ones, and the evidence,
for the controllers that estimate it, with the exact log p(d).

    python posterior.py                 # every case, on the cpu
    python posterior.py mala hmc        # some of them
    python posterior.py --gpu           # on the gpu
    python posterior.py --precision=float32   # in single precision, cpu or gpu
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
# the prior of the gaussian cases, and the supports of the wide and the tight uniform ones
SIGMA = 0.5
SUPPORT = (-5, 5)
TIGHT = (0, 1.2)

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

def crossfaded(base):
    """
    {base} with the linear model nested in a crossfade model, which a run then selects directly
    """
    head, rest = base.split("    model = altar.models.linear\n    model:\n", 1)
    block, tail = rest.split("    controller:\n", 1)
    nested = "".join(f"    {line}" if line.strip() else line for line in block.splitlines(True))
    nested = nested.replace("{linear.model.parameters}", "{linear.model.model.parameters}")
    return (f"{head}    model = altar.models.crossfade\n    model:\n"
            f"        model = altar.models.linear\n        model:\n{nested}    controller:\n{tail}")

def uniform(support, reparameterize=True, prep=None):
    """
    The settings of a uniform prior on {support}; a tight one also starts the chains from it
    """
    settings = [
        "--model.psets.all.prior=uniform",
        f"--model.psets.all.prior.support=({support[0]},{support[1]})",
        f"--model.psets.all.prior.reparameterize={reparameterize}",
    ]
    if prep or (prep is None and support == TIGHT):
        settings += [
            "--model.psets.all.prep=uniform",
            f"--model.psets.all.prep.support=({support[0]},{support[1]})",
        ]
    return settings

# the logistic-edged uniform prior on the wide support
SOFT = [
    "--model.psets.all.prior=softuniform",
    f"--model.psets.all.prior.support=({SUPPORT[0]},{SUPPORT[1]})",
]

# the prior of each case: the gaussian, the wide uniform or the tight uniform one
GAUSSIAN, WIDE, TIGHTLY = "gaussian", "wide", "tight"

# name: (the controller and sampler settings, the prior)
CASES = {
    "catmip": (["--controller=altar.bayesian.catmip", "--job.steps=256"], GAUSSIAN),
    "mcmc": (["--controller=altar.bayesian.mcmc", "--controller.rounds=16", "--job.steps=256"], GAUSSIAN),
    "catmip_hmc": (["--controller=altar.bayesian.catmip_hmc", "--job.steps=20"], GAUSSIAN),
    "hmc": (["--controller=altar.bayesian.hmc", "--job.steps=200"], GAUSSIAN),
    "catmip_mala": (["--controller=altar.bayesian.catmip_mala", "--job.steps=200"], GAUSSIAN),
    "mala": (["--controller=altar.bayesian.mala", "--job.steps=4000"], GAUSSIAN),
    "cf_catmip": (["--controller=altar.bayesian.cf_catmip", "--job.steps=256"], GAUSSIAN),
    "catmip-uniform": (["--controller=altar.bayesian.catmip", "--job.steps=256",
                        *uniform(SUPPORT, prep=True)], WIDE),
    "mcmc-uniform": (["--controller=altar.bayesian.mcmc", "--controller.rounds=16", "--job.steps=256",
                      *uniform(SUPPORT)], WIDE),
    "hmc-uniform": (["--controller=altar.bayesian.hmc", "--job.steps=200", *uniform(SUPPORT)], WIDE),
    "mala-uniform": (["--controller=altar.bayesian.mala", "--job.steps=4000", *uniform(SUPPORT)], WIDE),
    "catmip-tight": (["--controller=altar.bayesian.catmip", "--job.steps=256", *uniform(TIGHT)], TIGHTLY),
    "catmip-tight-physical": (["--controller=altar.bayesian.catmip", "--job.steps=256",
                               *uniform(TIGHT, reparameterize=False)], TIGHTLY),
    "cf_catmip-uniform": (["--controller=altar.bayesian.cf_catmip", "--job.steps=256", *uniform(SUPPORT)], WIDE),
    "cf_catmip-soft": (["--controller=altar.bayesian.cf_catmip", "--job.steps=256", *SOFT], WIDE),
    "cf_catmip-model": (["--controller=altar.bayesian.cf_catmip", "--job.steps=256"], GAUSSIAN),
    "cf_catmip-tight": (["--controller=altar.bayesian.cf_catmip", "--job.steps=256",
                         *uniform(TIGHT, reparameterize=False)], TIGHTLY),
}

# the tolerances, for 256 chains: the mean within this many posterior standard deviations,
# the standard deviations within this ratio, the correlations within this difference; 256
# independent exact draws reach 0.23, [0.83, 1.19] and 0.25 at worst over 200 trials
MEAN = 0.3
SD = (0.8, 1.25)
CORRELATION = 0.3
# and the evidence, for the controllers that estimate it, within this many nats: annealing from
# the prior, CATMIP's estimate is noisy, about a nat, and low with 256 chains; it also needs its
# initial samples drawn from the prior; cross-fading is exact here
EVIDENCE = 3.0


def exact(prior):
    """
    The mean and the covariance of the posterior, and the evidence log p(d), with the gaussian
    prior, with the wide uniform one, flat over the posterior, or with the tight uniform one
    """
    folder = EXAMPLES / CASE
    G = numpy.loadtxt(folder / "green.txt")
    d = numpy.loadtxt(folder / "data.txt")
    Cd = numpy.loadtxt(folder / "cd.txt")
    observations, parameters = G.shape
    W = numpy.linalg.inv(Cd)
    A = G.T @ W @ G
    if prior == GAUSSIAN:
        A += numpy.eye(parameters) / SIGMA**2
    cov = numpy.linalg.inv(A)
    mean = cov @ (G.T @ W @ d)
    if prior == GAUSSIAN:
        # log N(d; 0, C_d + σ² G G^T)
        C = Cd + SIGMA**2 * G @ G.T
        evidence = -0.5 * (d @ numpy.linalg.solve(C, d) + numpy.linalg.slogdet(C)[1]
                           + observations * numpy.log(2 * numpy.pi))
        return mean, cov, evidence
    # the integral of N(d; G θ, C_d) over θ, times the density of the prior, 1 / (b - a)
    low, high = SUPPORT if prior == WIDE else TIGHT
    r = d - G @ mean
    evidence = (-0.5 * (observations - parameters) * numpy.log(2 * numpy.pi)
                - 0.5 * numpy.linalg.slogdet(Cd)[1] + 0.5 * numpy.linalg.slogdet(cov)[1]
                - 0.5 * r @ W @ r - parameters * numpy.log(high - low))
    if prior == WIDE:
        return mean, cov, evidence
    # the draws of the posterior without a prior that land in the support, a fraction q of them,
    # which the integral over the support picks up
    total = 2_000_000
    draws = numpy.random.default_rng(1).multivariate_normal(mean, cov, size=total)
    draws = draws[((draws > TIGHT[0]) & (draws < TIGHT[1])).all(axis=1)]
    evidence += numpy.log(draws.shape[0] / total)
    return draws.mean(axis=0), numpy.cov(draws, rowvar=False), evidence


def samples(results):
    """
    The physical samples of the final step, and its evidence, if it has one
    """
    final = h5py.File(results / "step_final.h5")
    parameters = final["ParameterSets"]
    name = "all_physical" if "all_physical" in parameters else "all_sampling"
    evidence = final["Annealer"].get("log_evidence")
    return numpy.asarray(parameters[name]), None if evidence is None else float(numpy.asarray(evidence))


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
    settings, prior = CASES[name]
    scratch = pathlib.Path(tempfile.mkdtemp(prefix=f"altar-posterior-{name}-"))
    try:
        shutil.copytree(EXAMPLES / CASE, scratch / CASE)
        # the crossfade model selected directly, around the linear one
        base = crossfaded(BASE) if name == "cf_catmip-model" else BASE
        (scratch / "posterior.pfg").write_text(base)
        command = ["altar-linear", "--config=posterior.pfg", f"--job.gpus={int(gpu)}",
                   f"--job.precision={precision}", *settings]
        start = time.perf_counter()
        status = subprocess.run(command, cwd=scratch, capture_output=True, text=True, errors="replace")
        elapsed = time.perf_counter() - start
        if status.returncode != 0:
            (scratch / "run.log").write_text(status.stdout + status.stderr)
            keep = True
            return False, f"{name:22s} FAILED to run ({status.returncode}); see {scratch}/run.log"
        mean, cov, evidence = exact(prior=prior)
        theta, estimate = samples(scratch / "results")
        good, shift, low, high, drift = compare(theta, mean, cov)
        line = f"mean {shift:.2f} sd  sd ratio [{low:.2f}, {high:.2f}]  correlation {drift:.2f}"
        if estimate is not None:
            good = good and abs(estimate - evidence) <= EVIDENCE
            line += f"  log evidence {estimate:.2f} (exact {evidence:.2f})"
        verdict = "ok" if good else "FAIL"
        return good, f"{name:22s} {verdict:4s} {elapsed:6.0f}s  {line}"
    finally:
        if keep:
            print(f"  kept {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", nargs="*", help="the cases to run; all by default")
    parser.add_argument("--gpu", action="store_true", help="run on the gpu")
    parser.add_argument("--precision", default="float64", help="the precision, cpu or gpu")
    parser.add_argument("--keep", action="store_true", help="keep the scratch directories")
    parser.add_argument("--list", action="store_true", help="list the cases")
    options = parser.parse_args()

    if options.list:
        for name, (settings, prior) in CASES.items():
            print(f"{name:22s} {' '.join(settings)} ({prior} prior)")
        return 0
    if shutil.which("altar-linear") is None:
        print("altar-linear is not on the PATH")
        return 1
    unknown = [name for name in options.cases if name not in CASES]
    if unknown:
        print(f"unknown case(s): {', '.join(unknown)}; see --list")
        return 1

    print(f"tolerances: mean within {MEAN} sd, sd ratio in [{SD[0]}, {SD[1]}], "
          f"correlation within {CORRELATION}, log evidence within {EVIDENCE}")
    failures = 0
    for name in options.cases or CASES:
        good, line = run(name, gpu=options.gpu, precision=options.precision, keep=options.keep)
        print(line, flush=True)
        failures += not good
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


# end of file
