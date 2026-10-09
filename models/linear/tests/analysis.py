#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Check the analysis actions against the exact posterior of the linear example, patch-9,
N(μ, C), C = (F + I/σ²)⁻¹, F = Gᵀ C_d⁻¹ G. Over the posterior, the data log likelihood has the
mean log Z - (χ²(μ) + tr(F C)) / 2, with log Z the normalization, and the variance
tr((F C)²) / 2 + gᵀ C g, with g = Gᵀ C_d⁻¹ (d - G μ); the resolution matrix is R = C F.

A catmip run samples the posterior. The {forward} action measures its fit, with exact samples
drawn from N(μ, C) as its reference: the exact samples must match the moments above, and the
catmip samples the mean log likelihood of the exact ones. The {resolution} action computes F:
its tr R and its linearized standard deviations must be those of the exact posterior. The
{synthetic} action makes data for a checkerboard true model, whose noise must have χ²/N near 1;
the {recover} action, given exact samples of the posterior for those data, must find the z
scores of the truth under that posterior.

    python analysis.py
    python analysis.py --gpu
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy


# the example, relative to me
EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
CASE = "patch-9"
SIGMA = 0.5
SAMPLES = 4096
# the tolerances: of the means, in standard errors; of the variance, and of the resolution, relative
MEAN = 4
VARIANCE = 0.15
RESOLUTION = 1e-4
NOISE = (0.6, 1.5)
Z = 0.1

CONFIG = f"""
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
    controller = altar.bayesian.catmip
    controller:
        archiver = altar.bayesian.h5recorder
        archiver.output_dir = results
    job.tasks = 1
    job.chains = {SAMPLES}
    forward:
        theta = results/step_final.h5
        reference = exact.txt
        output = forward.h5
    resolution:
        theta = results/step_final.h5
        output = resolution.h5
    synthetic:
        theta = results/step_final.h5
        checkerboard = [all]
        grid = [6, 3]
        output = synthetic
    recover:
        theta = synthetic.txt
        truth = synthetic/truth.txt
        output = recover.h5
"""

# an action of the linear app, run by the altar plexus under the app's name
PLEXUS = "import altar, altar.models.linear; raise SystemExit(altar.shells.altar(name='linear').run())"


def exact(folder=EXAMPLES / CASE):
    """
    The mean and the covariance of the posterior for the data in {folder}, F, g and the
    normalization of the likelihood
    """
    G = numpy.loadtxt(folder / "green.txt")
    d = numpy.loadtxt(folder / "data.txt")
    Cd = numpy.loadtxt(folder / "cd.txt")
    W = numpy.linalg.inv(Cd)
    F = G.T @ W @ G
    cov = numpy.linalg.inv(F + numpy.eye(G.shape[1]) / SIGMA**2)
    mean = cov @ (G.T @ W @ d)
    r = d - G @ mean
    norm = -0.5 * (d.size * numpy.log(2 * numpy.pi) + numpy.linalg.slogdet(Cd)[1])
    return mean, cov, F, G.T @ W @ r, norm - 0.5 * r @ W @ r


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", action="store_true", help="run on the gpu")
    parser.add_argument("--keep", action="store_true", help="keep the scratch directory")
    options = parser.parse_args()
    gpus = f"--job.gpus={int(options.gpu)}"

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-analysis-"))
    try:
        shutil.copytree(EXAMPLES / CASE, scratch / CASE)
        (scratch / "linear.pfg").write_text(CONFIG)
        mean, cov, F, g, at_mean = exact()
        numpy.savetxt(scratch / "exact.txt",
                      numpy.random.default_rng(1).multivariate_normal(mean, cov, size=SAMPLES))
        for command in [["altar-linear", "--config=linear.pfg", gpus],
                        [sys.executable, "-c", PLEXUS, "forward", "--config=linear.pfg", gpus],
                        [sys.executable, "-c", PLEXUS, "resolution", "--config=linear.pfg", gpus],
                        [sys.executable, "-c", PLEXUS, "synthetic", "--config=linear.pfg", gpus],
                        "exact samples of the synthetic data",
                        [sys.executable, "-c", PLEXUS, "recover", "--config=linear.pfg", gpus]]:
            if isinstance(command, str):
                synthetic, synthetic_cov = exact(folder=scratch / "synthetic")[:2]
                numpy.savetxt(scratch / "synthetic.txt", numpy.random.default_rng(2)
                              .multivariate_normal(synthetic, synthetic_cov, size=SAMPLES))
                continue
            status = subprocess.run(command, cwd=scratch, capture_output=True, text=True,
                                    errors="replace")
            if status.returncode != 0:
                (scratch / "run.log").write_text(status.stdout + status.stderr)
                options.keep = True
                print(f"{command[0]} failed ({status.returncode}); see {scratch}/run.log")
                return 1
            print(status.stdout.split("journal (altar):")[-1].rstrip())

        with h5py.File(scratch / "forward.h5") as h5:
            catmip = numpy.asarray(h5["fit/samples/loglikelihood"])
            reference = numpy.asarray(h5["reference/fit/samples/loglikelihood"])
        with h5py.File(scratch / "resolution.h5") as h5:
            effective = float(numpy.asarray(h5["effective_parameters"]))
            linearized = numpy.asarray(h5["linearized_std"])
        with h5py.File(scratch / "recover.h5") as h5:
            z = numpy.asarray(h5["z"])
        truth = numpy.loadtxt(scratch / "synthetic" / "truth.txt")
        expected_z = (truth - synthetic) / numpy.sqrt(numpy.diag(synthetic_cov))
        G = numpy.loadtxt(EXAMPLES / CASE / "green.txt")
        Cd = numpy.loadtxt(EXAMPLES / CASE / "cd.txt")
        noise = numpy.loadtxt(scratch / "synthetic" / "data.txt") - G @ truth
        noise = noise @ numpy.linalg.solve(Cd, noise) / noise.size
        FC = F @ cov
        expected = at_mean - 0.5 * numpy.trace(FC)
        variance = 0.5 * numpy.trace(FC @ FC) + g @ cov @ g
        error = numpy.sqrt(variance / SAMPLES)
        checks = [
            ("exact samples, mean log L", reference.mean(), expected,
             abs(reference.mean() - expected) <= MEAN * error),
            ("exact samples, var log L", reference.var(), variance,
             abs(reference.var() / variance - 1) <= VARIANCE),
            ("catmip samples, mean log L", catmip.mean(), reference.mean(),
             abs(catmip.mean() - reference.mean()) <= MEAN * numpy.sqrt(2) * error),
            ("resolution, tr R", effective, numpy.trace(FC),
             abs(effective / numpy.trace(FC) - 1) <= RESOLUTION),
            ("resolution, linearized sd", numpy.abs(linearized / numpy.sqrt(numpy.diag(cov)) - 1).max(),
             0.0, numpy.abs(linearized / numpy.sqrt(numpy.diag(cov)) - 1).max() <= RESOLUTION),
            ("synthetic, chi^2/N of the noise", noise, 1.0, NOISE[0] <= noise <= NOISE[1]),
            ("recover, z of the truth", numpy.abs(z - expected_z).max(), 0.0,
             numpy.abs(z - expected_z).max() <= Z),
        ]
        failures = 0
        for name, value, target, good in checks:
            print(f"{name:32s} {'ok' if good else 'FAIL':4s} {value:12.4f}  (expected {target:.4f})")
            failures += not good
        return 1 if failures else 0
    finally:
        if options.keep:
            print(f"  kept {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())


# end of file
