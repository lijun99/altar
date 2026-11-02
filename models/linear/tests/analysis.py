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
scores of the truth under that posterior. The {optimize} action maximizes the posterior under
the conjugate prior, here the exact one: the best point of each of its methods must be μ, and
the standard deviations of Levenberg-Marquardt at the optimum those of C; with some parameters
fixed, Levenberg-Marquardt must find the mean and the standard deviations of the others under
the conditional posterior, and leave the fixed ones at their values. The {diagnose} action
compares the catmip posterior with the exact samples: their means and standard deviations must
agree; it also reports the diagnostics of the walks: every resampling keeps some parents, and
the last walk forgets most of its start.

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
SHIFT = 0.3
SD = (0.8, 1.25)
CORRELATION = 0.6
OPTIMUM = 1e-3
EXACT = 1e-6
LBFGS = 1e-2
# the parameters the conditional optimum holds fixed: three at a value, one at its start
FIXED = "[all[0:3]=0.1, all[10]]"

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
    diagnose:
        theta = results/step_final.h5
        reference = exact.txt
        output = diagnose.h5
    optimize:
        posterior = yes
        output = optimize.h5
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
                        [sys.executable, "-c", PLEXUS, "diagnose", "--config=linear.pfg", gpus],
                        [sys.executable, "-c", PLEXUS, "optimize", "--config=linear.pfg", gpus],
                        [sys.executable, "-c", PLEXUS, "optimize", "--config=linear.pfg", gpus,
                         "--linear.optimize.method=lm", "--linear.optimize.output=lm.h5"],
                        [sys.executable, "-c", PLEXUS, "optimize", "--config=linear.pfg", gpus,
                         "--linear.optimize.method=lbfgs", "--linear.optimize.output=lbfgs.h5"],
                        [sys.executable, "-c", PLEXUS, "optimize", "--config=linear.pfg", gpus,
                         "--linear.optimize.method=lm", f"--linear.optimize.fixed={FIXED}",
                         "--linear.optimize.output=fixed.h5"],
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
        with h5py.File(scratch / "diagnose.h5") as h5:
            shift = numpy.asarray(h5["reference/shift"]).max()
            ratio = numpy.asarray(h5["reference/sd_ratio"])
            final_beta = numpy.asarray(h5["schedule/beta"])[-1]
            parents = numpy.asarray(h5["schedule/parents"])[1:]
            correlation = numpy.asarray(h5["schedule/correlation"])[-1]
        with h5py.File(scratch / "optimize.h5") as h5:
            optimum = numpy.asarray(h5["best/theta"])
        with h5py.File(scratch / "lm.h5") as h5:
            lm, lm_sd = numpy.asarray(h5["best/theta"]), numpy.asarray(h5["best/sd"])
        with h5py.File(scratch / "lbfgs.h5") as h5:
            lbfgs = numpy.asarray(h5["best/theta"])
        with h5py.File(scratch / "fixed.h5") as h5:
            pinned, pinned_sd = numpy.asarray(h5["best/theta"]), numpy.asarray(h5["best/sd"])
            held, pinned_start = numpy.asarray(h5["fixed"]), numpy.asarray(h5["start/theta"])
        with h5py.File(scratch / "recover.h5") as h5:
            z = numpy.asarray(h5["z"])
        truth = numpy.loadtxt(scratch / "synthetic" / "truth.txt")
        expected_z = (truth - synthetic) / numpy.sqrt(numpy.diag(synthetic_cov))
        G = numpy.loadtxt(EXAMPLES / CASE / "green.txt")
        Cd = numpy.loadtxt(EXAMPLES / CASE / "cd.txt")
        noise = numpy.loadtxt(scratch / "synthetic" / "data.txt") - G @ truth
        noise = noise @ numpy.linalg.solve(Cd, noise) / noise.size
        sd = numpy.sqrt(numpy.diag(cov))
        offset = (numpy.abs(optimum - mean) / sd).max()
        lm_offset, lbfgs_offset = (numpy.abs(lm - mean) / sd).max(), (numpy.abs(lbfgs - mean) / sd).max()
        lm_sd_error = numpy.abs(lm_sd / sd - 1).max()
        # the conditional posterior of the free parameters, given the fixed ones at their values
        free = ~held
        values = numpy.zeros(held.size)
        values[:3] = 0.1
        values[10] = pinned_start[10]
        conditional_cov = numpy.linalg.inv(numpy.linalg.inv(cov)[numpy.ix_(free, free)])
        conditional = mean[free] + cov[numpy.ix_(free, held)] @ numpy.linalg.solve(
            cov[numpy.ix_(held, held)], values[held] - mean[held])
        conditional_sd = numpy.sqrt(numpy.diag(conditional_cov))
        fixed_offset = (numpy.abs(pinned[free] - conditional) / conditional_sd).max()
        fixed_sd_error = numpy.abs(pinned_sd[free] / conditional_sd - 1).max()
        fixed_held = numpy.abs(pinned[held] - values[held]).max() + numpy.abs(pinned_sd[held]).max()
        expected_held = numpy.zeros(held.size, dtype=bool)
        expected_held[[0, 1, 2, 10]] = True
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
            ("diagnose, final beta", final_beta, 1.0, final_beta == 1.0),
            ("diagnose, max |dmean|/sd", shift, 0.0, shift <= SHIFT),
            ("diagnose, sd ratio, worst", ratio[numpy.abs(ratio - 1).argmax()], 1.0,
             SD[0] <= ratio.min() and ratio.max() <= SD[1]),
            ("diagnose, parents, fewest", parents.min(), SAMPLES,
             0 < parents.min() and parents.max() <= SAMPLES),
            ("diagnose, final walk correlation", correlation, 0.0, 0 <= correlation <= CORRELATION),
            ("optimize, max |x - μ|/sd", offset, 0.0, offset <= OPTIMUM),
            ("optimize lm, max |x - μ|/sd", lm_offset, 0.0, lm_offset <= EXACT),
            ("optimize lm, sd, max rel error", lm_sd_error, 0.0, lm_sd_error <= EXACT),
            ("optimize lbfgs, max |x - μ|/sd", lbfgs_offset, 0.0, lbfgs_offset <= LBFGS),
            ("optimize fixed, the fixed ones", fixed_held, 0.0,
             (held == expected_held).all() and fixed_held == 0),
            ("optimize fixed, max |x - μ|/sd", fixed_offset, 0.0, fixed_offset <= EXACT),
            ("optimize fixed, sd, max rel error", fixed_sd_error, 0.0, fixed_sd_error <= EXACT),
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
