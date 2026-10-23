#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# slips sampled as a magnitude and a rake, the ensemble's {polar}: two static models on random
# green's functions, one cascaded, one annealed, naming the two components in opposite orders;
# their predictions and data likelihoods against numpy, from the components of the polar slips;
# and, with --gpu, the gradients of the ensemble against central differences of the likelihoods,
# with the magnitudes reparameterized
import pathlib
import sys
import tempfile
import numpy

PATCHES, CHAINS = 6, 8
MEMBERS = {"gps": (20, 0.1, "[across, parallel]", True), "insar": (15, 0.2, "[parallel, across]", False)}
CONFIG = """
slipmodel:
    model = altar.models.ensemble
    model:
        case = .
        polar = [magnitude, rake, parallel, across]
        psets_list = [magnitude, rake]
        psets:
            magnitude = contiguous
            rake = contiguous
            magnitude:
                count = {patches}
                prior = uniform
                prior.support = (0, 20)
                prior.reparameterize = True
            rake:
                count = {patches}
                prior = gaussian
                prior.mean = 0
                prior.sigma = 20
        models:
{members}
    job:
        tasks = 1
        gpus = {gpus}
        chains = {chains}
        precision = float64
        gpuprecision = float64
"""
MEMBER = """            {name} = altar.models.seismic.static
            {name}:
                cascaded = {cascaded}
                case = .
                patches = {patches}
                psets_list = {order}
                green = {name}.gf.h5
                dataobs:
                    observations = {observations}
                    data_file = {name}.data.h5
                    cd_std = {sigma}"""


def components(θ):
    m, r = θ[:, :PATCHES], numpy.radians(θ[:, PATCHES:])
    return {"parallel": m * numpy.cos(r), "across": m * numpy.sin(r)}


def main():
    gpu = "--gpu" in sys.argv
    import h5py
    rng = numpy.random.default_rng(3)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-polar-"))
    G, d = {}, {}
    for name, (observations, sigma, order, cascaded) in MEMBERS.items():
        G[name] = rng.standard_normal((observations, 2 * PATCHES))
        d[name] = rng.standard_normal(observations)
        with h5py.File(scratch / f"{name}.gf.h5", "w") as h5:
            h5[f"{name}.gf"] = G[name]
        with h5py.File(scratch / f"{name}.data.h5", "w") as h5:
            h5[f"{name}.data"] = d[name]
    members = "\n".join(MEMBER.format(name=name, cascaded=cascaded, patches=PATCHES, order=order,
                                      observations=observations, sigma=sigma)
                        for name, (observations, sigma, order, cascaded) in MEMBERS.items())
    (scratch / "slipmodel.pfg").write_text(CONFIG.format(patches=PATCHES, members=members, gpus=int(gpu), chains=CHAINS))

    import os
    os.chdir(scratch)
    sys.argv = [sys.argv[0], "--config=slipmodel.pfg"]
    import altar
    import altar.models.seismic
    app = altar.shells.altar(name="slipmodel").initialize()
    model = app.model

    θ = numpy.column_stack([rng.uniform(1, 10, (CHAINS, PATCHES)), rng.normal(0, 30, (CHAINS, PATCHES))])

    def member_columns(name):
        order = MEMBERS[name][2].strip("[]").replace(" ", "").split(",")
        c = components(θ)
        return numpy.column_stack([c[order[0]], c[order[1]]])

    def loglike(name, x):
        # the data log likelihood of model {name} at the polar slips {x}, up to a constant
        order = MEMBERS[name][2].strip("[]").replace(" ", "").split(",")
        c = components(x)
        r = (G[name] @ numpy.column_stack([c[order[0]], c[order[1]]]).T).T - d[name]
        return -0.5 * ((r / MEMBERS[name][1]) ** 2).sum(axis=1)

    failures = 0

    def check(label, error, tolerance):
        nonlocal failures
        ok = error <= tolerance
        failures += not ok
        print(f"  {label}: {error:.2e} (tolerance {tolerance:.0e}) {'ok' if ok else 'FAILED'}")

    # the predictions and the parameters of each model
    predictions = model.forward_problem(application=app, theta=θ)
    for name in MEMBERS:
        expected = (G[name] @ member_columns(name).T).T
        check(f"{name}: predictions", numpy.abs(predictions[f"{name}.data"] - expected).max() / numpy.abs(expected).max(), 1e-12)
        check(f"{name}: member_theta", numpy.abs(model.member_theta(name, θ) - member_columns(name)).max(), 1e-12)
    # the data likelihoods, up to their constant
    for name, member in model.models.items():
        if gpu:
            import altar.cuda
            likelihood = altar.cuda.vector(shape=CHAINS, dtype="float64").zero()
            member.eval_data_likelihood(theta=model._member_theta(name=name, theta=altar.cuda.matrix(source=θ, dtype="float64"), batch=CHAINS),
                                        likelihood=likelihood)
            likelihood = numpy.array(likelihood)
        else:
            likelihood = numpy.zeros(CHAINS)
            member.eval_data_likelihood(theta=model._member_theta(name=name, theta=θ, batch=CHAINS), likelihood=likelihood)
        expected = loglike(name, θ)
        check(f"{name}: data likelihood", numpy.abs((likelihood - likelihood[0]) - (expected - expected[0])).max() / numpy.abs(expected).max(), 1e-10)
    # the jacobian of the parameters of each model, against central differences
    h = 1e-6
    for name in MEMBERS:
        D = model.member_jacobian(name, θ[0])
        fd = numpy.zeros_like(D)
        for i in range(θ.shape[1]):
            step = numpy.zeros(θ.shape[1])
            step[i] = h
            fd[i] = (model.member_theta(name, θ[0] + step)[0] - model.member_theta(name, θ[0] - step)[0]) / (2 * h)
        check(f"{name}: member_jacobian", numpy.abs(D - fd).max(), 1e-7)

    if gpu:
        import altar.cuda

        class Step:
            pass

        step = Step()
        step.theta = altar.cuda.matrix(source=θ, dtype="float64")
        step.prior_gradient = altar.cuda.matrix(shape=θ.shape, dtype="float64").zero()
        step.data_gradient = altar.cuda.matrix(shape=θ.shape, dtype="float64").zero()
        step.Jacobian = altar.cuda.matrix(source=numpy.ones(θ.shape), dtype="float64")
        model.gradient(controller=None, step=step)
        prior_gradient, data_gradient = numpy.array(step.prior_gradient), numpy.array(step.data_gradient)

        def central(name):
            g = numpy.zeros_like(θ)
            for i in range(θ.shape[1]):
                step = numpy.zeros(θ.shape[1])
                step[i] = h
                g[:, i] = (loglike(name, θ + step) - loglike(name, θ - step)) / (2 * h)
            return g

        # the annealed model's: its data gradient, in physical space
        expected = central("insar")
        check("insar: data gradient", numpy.abs(data_gradient - expected).max() / numpy.abs(expected).max(), 1e-7)
        # the cascaded model's: in the prior gradient, with the Jacobian of the reparameterized
        # magnitudes, after the priors: the logit's 1 - 2s for the magnitudes, -rake/20^2
        s = θ[:, :PATCHES] / 20
        jacobian = numpy.column_stack([20 * s * (1 - s), numpy.ones((CHAINS, PATCHES))])
        priors = numpy.column_stack([1 - 2 * s, -θ[:, PATCHES:] / 400])
        expected = priors + jacobian * central("gps")
        check("gps: cascaded gradient", numpy.abs(prior_gradient - expected).max() / numpy.abs(expected).max(), 1e-7)

    print("polar slips:", "passed" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


# end of file
