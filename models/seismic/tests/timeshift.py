#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the time shifts of the stations in the kinematic model on triangles, on a small planar fault with
# random green's functions: the predictions shifted by whole samples against the unshifted ones,
# by fractions against the cpu reference, and the gradient, the shifts included, against central
# differences of the likelihood; on the cpu, or with --gpu on cuda
import os
import pathlib
import sys
import tempfile
import numpy

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trimesh"))

STATIONS, TRACES, LENGTH, NT, SIGMA, CHAINS = 5, 3, 20, 40, 1e-2, 6
CONFIG = """
slipmodel:
    model = altar.models.seismic.kinematicmesh
    model:
        case = .
        mesh = mesh.h5
        Nmesh = 6
        patches = {patches}
        green = kinematicG.gf.h5
        dataobs:
            observations = {observations}
            data_file = kinematicG.data.h5
            cd_std = {sigma}
        Nt = {nt}
        dt = 1.0
        shifts = {stations}
        traces = {traces}
        psets_list = [strikeslip, dipslip, risetime, rupturevelocity, hypocenter, shift]
        psets:
            strikeslip = contiguous
            dipslip = contiguous
            risetime = contiguous
            rupturevelocity = contiguous
            hypocenter = contiguous
            shift = contiguous
            strikeslip:
                count = {patches}
                prior = gaussian
                prior.sigma = 0.5
            dipslip:
                count = {patches}
                prior = gaussian
                prior.mean = 2
                prior.sigma = 1
            risetime:
                count = {patches}
                prior = gaussian
                prior.mean = 5
                prior.sigma = 1
            rupturevelocity:
                count = {patches}
                prior = gaussian
                prior.mean = 3
                prior.sigma = 0.5
            hypocenter:
                count = 2
                prior = gaussian
                prior.mean = 15
                prior.sigma = 5
            shift:
                count = {stations}
                prior = gaussian
                prior.sigma = 2
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
    from kinematic_mesh import fault
    rng = numpy.random.default_rng(9)
    xyz, uv, faces, sources, owners = fault("planar")
    Np = len(faces)
    observations = STATIONS * TRACES * LENGTH
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-timeshift-"))
    with h5py.File(scratch / "mesh.h5", "w") as h5:
        h5["vertices"], h5["uv"], h5["faces"] = xyz, uv, faces
        h5["sources/uv"], h5["sources/patch"] = sources, owners
    # smooth green's functions, so that the traces look like bandpassed ones
    t = numpy.arange(LENGTH)
    G = numpy.stack([numpy.sin(2 * numpy.pi * (t / rng.uniform(6, 15) + rng.uniform())) for _ in range(2 * Np * NT * STATIONS * TRACES)])
    G = G.reshape(2 * Np * NT, STATIONS * TRACES * LENGTH) * 1e-2
    with h5py.File(scratch / "kinematicG.gf.h5", "w") as h5:
        h5["kinematicG.gf"] = G
    with h5py.File(scratch / "kinematicG.data.h5", "w") as h5:
        h5["kinematicG.data"] = rng.standard_normal(observations) * 1e-2
    (scratch / "slipmodel.pfg").write_text(CONFIG.format(patches=Np, observations=observations, sigma=SIGMA, nt=NT,
                                                         stations=STATIONS, traces=TRACES, gpus=int(gpu), chains=CHAINS))
    os.chdir(scratch)
    sys.argv = [sys.argv[0], "--config=slipmodel.pfg"]
    import altar
    import altar.models.seismic
    from altar.models.seismic import shifts as timeshifts
    app = altar.shells.altar(name="slipmodel").initialize()
    model = app.model

    failures = 0

    def check(label, error, tolerance):
        nonlocal failures
        ok = error <= tolerance
        failures += not ok
        print(f"  {label}: {error:.2e} (tolerance {tolerance:.0e}) {'ok' if ok else 'FAILED'}")

    θ = numpy.column_stack([rng.normal(0, 0.3, (CHAINS, Np)), rng.uniform(1, 3, (CHAINS, Np)),
                            rng.uniform(4, 6, (CHAINS, Np)), rng.uniform(2.5, 3.5, (CHAINS, Np)),
                            rng.uniform(12, 18, (CHAINS, 2)), rng.uniform(-3, 3, (CHAINS, STATIONS))])
    S = θ.shape[1] - STATIONS
    unshifted = θ.copy()
    unshifted[:, S:] = 0
    d0 = model.forward_problem(application=app, theta=unshifted)["data"]
    # whole samples: the unshifted traces moved, zeros in front
    whole = unshifted.copy()
    whole[:, S:] = numpy.arange(1, STATIONS + 1)
    d = model.forward_problem(application=app, theta=whole)["data"].reshape(CHAINS, STATIONS, TRACES, LENGTH)
    expected = numpy.zeros_like(d)
    r0 = d0.reshape(d.shape)
    for k in range(STATIONS):
        expected[:, k, :, k + 1:] = r0[:, k, :, :LENGTH - k - 1]
    check("whole-sample shifts", numpy.abs(d - expected).max() / numpy.abs(expected).max(), 1e-12)
    # fractions of samples, against the cpu reference
    d = model.forward_problem(application=app, theta=θ)["data"]
    reference = timeshifts.shift(model.forward_problem(application=app, theta=unshifted)["data"], θ[:, S:], TRACES, LENGTH)
    check("fractional shifts", numpy.abs(d - reference).max() / numpy.abs(reference).max(), 1e-12)

    # the gradient of the data log likelihood against central differences
    observed = model.dataobs.observed()
    loglike = lambda x: -0.5 * (((model.forward_problem(application=app, theta=x)["data"] - observed) / SIGMA) ** 2).sum(axis=1)

    class Step:
        pass

    step = Step()
    if gpu:
        import altar.cuda
        step.theta = altar.cuda.matrix(source=θ, dtype="float64")
        step.prior_gradient = altar.cuda.matrix(shape=θ.shape, dtype="float64").zero()
        step.data_gradient = altar.cuda.matrix(shape=θ.shape, dtype="float64").zero()
        model.gradient(controller=None, step=step)
        gradient = numpy.array(step.data_gradient)
    else:
        step.theta, step.grad_prior, step.grad_data = θ.copy(), numpy.zeros_like(θ), numpy.zeros_like(θ)
        model.gradient(controller=None, step=step)
        gradient = step.grad_data
    h = 1e-6
    fd = numpy.zeros_like(θ)
    for i in range(θ.shape[1]):
        e = numpy.zeros(θ.shape[1])
        e[i] = h
        fd[:, i] = (loglike(θ + e) - loglike(θ - e)) / (2 * h)
    columns = slice(S, None)
    check("gradient, the shifts", numpy.abs(gradient[:, columns] - fd[:, columns]).max() / numpy.abs(fd[:, columns]).max(), 1e-6)
    check("gradient, the slips", numpy.abs(gradient[:, :2 * Np] - fd[:, :2 * Np]).max() / numpy.abs(fd[:, :2 * Np]).max(), 1e-6)
    check("gradient, all", numpy.abs(gradient - fd).max() / numpy.abs(fd).max(), 1e-5)

    print("time shifts:", "passed" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


# end of file
