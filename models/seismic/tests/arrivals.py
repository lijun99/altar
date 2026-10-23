#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the arrival times from the hypocenter, altar.models.seismic.arrivals, on a planar mesh dipping 30
# degrees: the times against straight rays from the hypocenter's known position, the gradient against
# central differences of the likelihood, and a table of travel times against the straight rays; with
# --gpu, on the cuda backend
import os
import pathlib
import sys
import tempfile
import numpy

NS, ND, SIZE, DIP, VELOCITY, STATIONS, CHAINS = 4, 3, 10.0, 30.0, 6.0, 12, 8
CONFIG = """
slipmodel:
    model = altar.models.seismic.arrivals
    model:
        case = .
        mesh = mesh.h5
        stations = stations.txt
        velocity = {velocity}
{table}        dataobs:
            observations = {stations}
            data_file = picks.h5
            cd_std = 0.2
        psets_list = [hypo_strike, hypo_dip]
        psets:
            hypo_strike = contiguous
            hypo_dip = contiguous
            hypo_strike:
                count = 1
                prior = gaussian
                prior.mean = 20
                prior.sigma = 5
            hypo_dip:
                count = 1
                prior = gaussian
                prior.mean = 15
                prior.sigma = 5
    job:
        tasks = 1
        gpus = {gpus}
        chains = {chains}
        precision = float64
        gpuprecision = float64
"""


def main():
    gpu = "--gpu" in sys.argv
    table = "--table" in sys.argv
    import h5py
    rng = numpy.random.default_rng(5)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-arrivals-"))
    # a planar mesh dipping DIP degrees, its surface coordinates exact
    u, v = numpy.arange(NS + 1) * SIZE, numpy.arange(ND + 1) * SIZE
    U, V = numpy.meshgrid(u, v, indexing="ij")
    uv = numpy.column_stack([U.ravel(), V.ravel()])
    δ = numpy.radians(DIP)
    place = lambda uv: numpy.column_stack([uv[:, 1] * numpy.cos(δ), uv[:, 0], 5.0 + uv[:, 1] * numpy.sin(δ)])
    index = lambda i, j: i * (ND + 1) + j
    faces = numpy.array([f for i in range(NS) for j in range(ND)
                         for f in [(index(i, j), index(i + 1, j), index(i + 1, j + 1)),
                                   (index(i, j), index(i + 1, j + 1), index(i, j + 1))]])
    with h5py.File(scratch / "mesh.h5", "w") as h5:
        h5["vertices"], h5["uv"], h5["faces"] = place(uv), uv, faces
    stations = numpy.column_stack([rng.uniform(-30, 60, STATIONS), rng.uniform(-20, 60, STATIONS), rng.uniform(0, 2, STATIONS)])
    numpy.savetxt(scratch / "stations.txt", stations)
    with h5py.File(scratch / "picks.h5", "w") as h5:
        h5["picks"] = rng.uniform(2, 10, STATIONS)
    entry = ""
    if table:
        depths, distances = numpy.linspace(-3, 30, 200), numpy.linspace(0, 120, 600)
        with h5py.File(scratch / "times.h5", "w") as h5:
            h5["depths"], h5["distances"] = depths, distances
            h5["times"] = numpy.hypot(depths[:, None], distances[None, :]) / VELOCITY
        entry = "        table = times.h5\n"
    (scratch / "slipmodel.pfg").write_text(CONFIG.format(velocity=VELOCITY, table=entry, stations=STATIONS,
                                                         gpus=int(gpu), chains=CHAINS))
    os.chdir(scratch)
    sys.argv = [sys.argv[0], "--config=slipmodel.pfg"]
    import altar
    import altar.models.seismic
    app = altar.shells.altar(name="slipmodel").initialize()
    model = app.model

    failures = 0

    def check(label, error, tolerance):
        nonlocal failures
        ok = error <= tolerance
        failures += not ok
        print(f"  {label}: {error:.2e} (tolerance {tolerance:.0e}) {'ok' if ok else 'FAILED'}")

    θ = numpy.column_stack([rng.uniform(2, 38, CHAINS), rng.uniform(2, 28, CHAINS)])
    # the times against straight rays from the known position of each hypocenter
    times = model.forward_problem(application=app, theta=θ)["data"]
    x = place(θ)
    surface = numpy.column_stack([stations[:, :2], -stations[:, 2]])
    expected = numpy.sqrt(((x[:, None, :] - surface[None]) ** 2).sum(axis=2)) / VELOCITY
    check("times" + (", from the table" if table else ""), numpy.abs(times - expected).max(), 1e-4 if table else 1e-12)
    # the gradient against central differences of the data log likelihood
    observed = model.dataobs.observed()
    loglike = lambda θ: -0.5 * (((model.forward_problem(application=app, theta=θ)["data"] - observed) / 0.2) ** 2).sum(axis=1)

    class Step:
        pass

    step = Step()
    if gpu:
        import altar.cuda
        new = lambda a: altar.cuda.matrix(source=a, dtype="float64")
    else:
        new = lambda a: numpy.array(a, dtype=float)
    step.theta, step.prior_gradient, step.data_gradient = new(θ), new(numpy.zeros_like(θ)), new(numpy.zeros_like(θ))
    model.gradient(controller=None, step=step)
    h = 1e-6
    fd = numpy.column_stack([(loglike(θ + e) - loglike(θ - e)) / (2 * h) for e in numpy.identity(2) * h])
    check("data gradient", numpy.abs(numpy.array(step.data_gradient) - fd).max() / numpy.abs(fd).max(), 1e-6)
    prior = numpy.column_stack([(20 - θ[:, 0]) / 25, (15 - θ[:, 1]) / 25])
    check("prior gradient", numpy.abs(numpy.array(step.prior_gradient) - prior).max(), 1e-12)

    print("arrival times:", "passed" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


# end of file
