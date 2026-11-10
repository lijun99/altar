#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the noise scales of the ensemble's {noise}: two static models on random green's functions, one
# cascaded with a full covariance, one annealed with a constant one and masked observations, each
# with its covariance scaled by exp(2h); their likelihoods and the residuals of the optimize
# action against numpy; and, with --gpu, the gradients of the ensemble against central differences
# of the likelihoods, by the slips and by the noise scales
import math
import pathlib
import sys
import tempfile
import numpy

PATCHES, CHAINS, SIGMA = 4, 8, 10.0
CONFIG = """
slipmodel:
    model = altar.models.ensemble
    model:
        case = .
        psets_list = [parallel, h_gps, across, h_insar]
        noise:
            gps = h_gps
            insar = h_insar
        psets:
            parallel = contiguous
            across = contiguous
            h_gps = contiguous
            h_insar = contiguous
            parallel:
                count = {patches}
                prior = gaussian
                prior.mean = 0
                prior.sigma = {sigma}
            across:
                count = {patches}
                prior = gaussian
                prior.mean = 0
                prior.sigma = {sigma}
            h_gps:
                count = 1
                prior = gaussian
                prior.mean = 0
                prior.sigma = 1
            h_insar:
                count = 1
                prior = gaussian
                prior.mean = 0
                prior.sigma = 1
        models:
            gps = altar.models.seismic.static
            gps:
                cascaded = True
                case = .
                patches = {patches}
                psets_list = [parallel, across]
                green = gps.gf.h5
                dataobs:
                    observations = {gps}
                    data_file = gps.data.h5
                    cd_file = gps.cd.h5
            insar = altar.models.seismic.static
            insar:
                case = .
                patches = {patches}
                psets_list = [parallel, across]
                green = insar.gf.h5
                dataobs:
                    observations = {insar}
                    data_file = insar.data.h5
                    mask_dataset = mask
                    cd_std = 0.2
    job:
        tasks = 1
        gpus = {gpus}
        chains = {chains}
        precision = float64
        gpuprecision = float64
"""


def main():
    gpu = "--gpu" in sys.argv
    import h5py
    rng = numpy.random.default_rng(5)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-noise-"))
    size = {"gps": 12, "insar": 15}
    G = {name: rng.standard_normal((n, 2 * PATCHES)) for name, n in size.items()}
    d = {name: rng.standard_normal(n) for name, n in size.items()}
    A = rng.standard_normal((size["gps"], size["gps"]))
    C = {"gps": 0.05 * (A @ A.T / size["gps"] + numpy.identity(size["gps"])),
         "insar": 0.2 ** 2 * numpy.identity(size["insar"])}
    mask = {"gps": numpy.ones(size["gps"], dtype=bool), "insar": rng.uniform(size=size["insar"]) > 0.3}
    for name in size:
        with h5py.File(scratch / f"{name}.gf.h5", "w") as h5:
            h5[f"{name}.gf"] = G[name]
        with h5py.File(scratch / f"{name}.data.h5", "w") as h5:
            h5[f"{name}.data"] = numpy.where(mask[name], d[name], numpy.nan)
            h5["mask"] = mask[name].astype(numpy.int8)
    with h5py.File(scratch / "gps.cd.h5", "w") as h5:
        h5["gps.cd"] = C["gps"]
    (scratch / "slipmodel.pfg").write_text(CONFIG.format(
        patches=PATCHES, sigma=SIGMA, gps=size["gps"], insar=size["insar"], gpus=int(gpu), chains=CHAINS))

    import os
    os.chdir(scratch)
    sys.argv = [sys.argv[0], "--config=slipmodel.pfg"]
    import altar
    import altar.models.seismic
    app = altar.shells.altar(name="slipmodel").initialize()
    model = app.model

    # theta: parallel, h_gps, across, h_insar
    θ = numpy.column_stack([rng.normal(0, 2, (CHAINS, PATCHES)), rng.normal(0, 0.7, CHAINS),
                            rng.normal(0, 2, (CHAINS, PATCHES)), rng.normal(0, 0.7, CHAINS)])
    noise = {"gps": PATCHES, "insar": 2 * PATCHES + 1}
    slips = numpy.r_[0:PATCHES, PATCHES + 1:2 * PATCHES + 1]

    def loglike(name, x):
        # the data log likelihood of model {name} at {x}, with its covariance scaled by exp(2h)
        h = x[:, noise[name]]
        keep = mask[name]
        r = ((G[name] @ x[:, slips].T).T - d[name])[:, keep]
        Cm = C[name][numpy.ix_(keep, keep)]
        chi2 = numpy.einsum("si,ij,sj->s", r, numpy.linalg.inv(Cm), r)
        n = int(keep.sum())
        return -0.5 * (n * math.log(2 * math.pi) + numpy.linalg.slogdet(Cm)[1]) - n * h - 0.5 * numpy.exp(-2 * h) * chi2

    def logprior(x):
        return -0.5 * (x[:, slips] ** 2).sum(axis=1) / SIGMA ** 2 - 0.5 * (x[:, list(noise.values())] ** 2).sum(axis=1)

    failures = 0

    def check(label, error, tolerance):
        nonlocal failures
        ok = error <= tolerance
        failures += not ok
        print(f"  {label}: {error:.2e} (tolerance {tolerance:.0e}) {'ok' if ok else 'FAILED'}")

    # the likelihoods of a step: the cascaded model's in the prior, the other's in the data
    if gpu:
        import altar.cuda
        array = lambda x: altar.cuda.matrix(source=x, dtype="float64") if numpy.ndim(x) == 2 \
            else altar.cuda.vector(source=x, dtype="float64")
    else:
        array = lambda x: numpy.array(x, dtype=float)

    class Dispatcher:
        def __getattr__(self, name):
            return name

        def notify(self, **kwds):
            return

    class Annealer:
        dispatcher = Dispatcher()

    class Step:
        pass

    step = Step()
    step.beta = 1.0
    step.theta = array(θ)
    step.prior, step.data, step.posterior = (array(numpy.zeros(CHAINS)) for _ in range(3))
    model.likelihoods(annealer=Annealer(), step=step)
    prior, data = numpy.array(step.prior), numpy.array(step.data)
    prior -= logprior(θ) - 0.5 * (2 * PATCHES * math.log(2 * math.pi * SIGMA ** 2) + 2 * math.log(2 * math.pi))
    check("gps: cascaded likelihood", numpy.abs(prior - loglike("gps", θ)).max(), 1e-9)
    check("insar: data likelihood", numpy.abs(data - loglike("insar", θ)).max(), 1e-9)

    # the residuals of the optimize action, with the noise scales held at the start's
    from altar.actions.Optimize import Problem
    free = numpy.ones(θ.shape[1], dtype=bool)
    try:
        Problem(plexus=app, model=model, start=θ[0], free=free, posterior=False, prior=None,
                batch=None, steps=None, lower=None, upper=None)
        check("optimize: a free noise scale is refused", 1, 0)
    except ValueError:
        check("optimize: a free noise scale is refused", 0, 0)
    free[list(noise.values())] = False
    problem = Problem(plexus=app, model=model, start=θ[0], free=free, posterior=False, prior=None,
                      batch=None, steps=None, lower=None, upper=None)
    rows = θ.copy()
    rows[:, list(noise.values())] = θ[0, list(noise.values())]
    values = problem.constant - 0.5 * (problem.evaluate(theta=rows, with_prior=False) ** 2).sum(axis=1)
    expected = loglike("gps", rows) + loglike("insar", rows)
    check("optimize: objective", numpy.abs(values - expected).max() / numpy.abs(expected).max(), 1e-12)

    if gpu:
        step.prior_gradient = altar.cuda.matrix(shape=θ.shape, dtype="float64").zero()
        step.data_gradient = altar.cuda.matrix(shape=θ.shape, dtype="float64").zero()
        step.Jacobian = altar.cuda.matrix(source=numpy.ones(θ.shape), dtype="float64")
        model.gradient(controller=None, step=step)
        prior_gradient, data_gradient = numpy.array(step.prior_gradient), numpy.array(step.data_gradient)

        def central(f):
            g = numpy.zeros_like(θ)
            for i in range(θ.shape[1]):
                e = numpy.zeros(θ.shape[1])
                e[i] = 1e-6
                g[:, i] = (f(θ + e) - f(θ - e)) / 2e-6
            return g

        expected = central(lambda x: loglike("insar", x))
        check("insar: data gradient", numpy.abs(data_gradient - expected).max() / numpy.abs(expected).max(), 1e-7)
        expected = central(lambda x: logprior(x) + loglike("gps", x))
        check("gps: cascaded gradient", numpy.abs(prior_gradient - expected).max() / numpy.abs(expected).max(), 1e-7)

    print("noise scales:", "passed" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


# end of file
