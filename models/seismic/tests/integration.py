#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the source time functions integrated over the time intervals, model.integration = step or bin,
# on the cpu, with rise times from a tenth of dt to 5 dt: on triangles, through SlipHistory, the
# slips of each interval against quadrature of the slip rates at the arrival times of the source
# points, for the triangles peaking at 0.25, 0.5 and 0.8 and the raised cosine; the slips in all
# against those of the patches, on triangles and on rectangles, where sampling falls short
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings
import numpy
from scipy.integrate import IntegrationWarning, quad

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trimesh"))

NT, DT, SAMPLES = 40, 1.0, 3
CASES = [("triangle", 0.25), ("triangle", 0.5), ("triangle", 0.8), ("cosine", 0.5)]
RECTANGLE = """
slipmodel:
    model = altar.models.seismic.kinematic
    model:
        case = ..
        Nas = 3
        Ndd = 2
        Nmesh = 6
        dsp = 10.0
        Nt = {nt}
        Npt = 2
        dt = {dt}
        stf = {stf}
        peak = {peak}
        integration = {integration}
        green = kinematicG.gf.h5
        dataobs:
            observations = 10
            data_file = kinematicG.data.h5
            cd_std = 1
        psets_list = [strikeslip, dipslip, risetime, rupturevelocity, hypocenter]
        psets:
            strikeslip = contiguous
            dipslip = contiguous
            risetime = contiguous
            rupturevelocity = contiguous
            hypocenter = contiguous
            strikeslip.count = 6
            dipslip.count = 6
            risetime.count = 6
            rupturevelocity.count = 6
            hypocenter.count = 2
    job:
        tasks = 1
        gpus = 0
        chains = {samples}
        precision = float64
"""


def rate(u, Tr, stf, peak):
    """
    The slip rate {u} after the onset, of unit area over the rise time {Tr}
    """
    if u <= 0 or u >= Tr:
        return 0.0
    if stf == "cosine":
        return (1 - numpy.cos(2 * numpy.pi * u / Tr)) / Tr
    if u < peak * Tr:
        return 2 * u / (peak * Tr * Tr)
    return 2 * (Tr - u) / ((1 - peak) * Tr * Tr)


def cumulative(s, Tr, stf, peak):
    """
    By quadrature: V, the slip by {s} after the onset, and W, the integral of V up to {s}
    """
    if s <= 0:
        return 0.0, 0.0
    end = min(s, Tr)
    points = [peak * Tr] if stf == "triangle" and peak * Tr < end else None
    opts = dict(points=points, epsabs=1e-15, epsrel=1e-14, limit=200)
    V = quad(rate, 0, end, args=(Tr, stf, peak), **opts)[0]
    # W(s) = int_0^s v(u) (s - u) du
    W = quad(lambda u: rate(u, Tr, stf, peak) * (s - u), 0, end, **opts)[0]
    return V, W


def reference(arrivals, risetime, slip, stf, peak, dt, Nt):
    """
    Mb (Nt x 2 x patches) of step and bin from quadrature, at the {arrivals} of the source points,
    stored patch by patch
    """
    Np = risetime.size
    points = arrivals.reshape(Np, -1)
    # the edges of the intervals, one more before for bin
    edges = dt * numpy.arange(-1, Nt + 2)
    step = numpy.zeros((Nt, Np))
    binned = numpy.zeros((Nt, Np))
    for p in range(Np):
        for T in points[p]:
            VW = numpy.array([cumulative(t - T, risetime[p], stf, peak) for t in edges])
            V, W = VW[:, 0], VW[:, 1]
            step[:, p] += numpy.diff(V)[1:Nt + 1]
            binned[:, p] += (W[2:] - 2 * W[1:-1] + W[:-2])[:Nt] / dt
        step[:, p] /= points.shape[1]
        binned[:, p] /= points.shape[1]
    rakes = lambda c: c[:, None, :] * slip.T[None, :, :]
    return rakes(step), rakes(binned)


def triangles(check, rng):
    """
    The slip histories on triangles against quadrature, and the slips in all
    """
    from kinematic_mesh import fault
    from altar.models.seismic.history import SlipHistory
    xyz, uv, faces, sources, owners = fault("planar")
    Np = len(faces)
    slip = numpy.stack([rng.normal(0, 0.3, (SAMPLES, Np)), rng.uniform(1, 3, (SAMPLES, Np))], axis=-1)
    risetime = rng.uniform(0.1, 5, (SAMPLES, Np)) * DT
    velocity = rng.uniform(2.0, 3.5, (SAMPLES, Np))
    hypocenter = rng.uniform(12, 18, (SAMPLES, 2))
    for stf, peak in CASES:
        make = lambda mode, Nt: SlipHistory(xyz, faces, uv, sources, owners, Nt=Nt, dt=DT, Nmesh=6,
                                            stf=stf, peak=peak, integration=mode)
        arrivals, _ = make("sample", NT).arrivals(velocity, hypocenter)
        step, binned = make("step", NT), make("bin", NT)
        worst = {"step": 0.0, "bin": 0.0}
        for k in range(SAMPLES):
            args = (slip[k], risetime[k], velocity[k], hypocenter[k])
            expected = reference(arrivals[k], risetime[k], slip[k], stf, peak, DT, NT)
            for mode, history, want in (("step", step, expected[0]), ("bin", binned, expected[1])):
                error = numpy.abs(history(*args) - want).max() / numpy.abs(want).max()
                worst[mode] = max(worst[mode], error)
        for mode in ("step", "bin"):
            check(f"triangles, {stf} {peak:g}, {mode}: Mb against quadrature", worst[mode], 1e-11)
        # the slips in all, over enough intervals
        steps = int(numpy.ceil((arrivals.max() + risetime.max()) / DT)) + 2
        for mode in ("sample", "step", "bin"):
            total = make(mode, steps)(slip, risetime, velocity, hypocenter).sum(axis=1)
            error = numpy.abs(total - numpy.moveaxis(slip, -1, 1)).max() / numpy.abs(slip).max()
            if mode == "sample":
                print(f"  triangles, {stf} {peak:g}, sample: the slips in all off by {error:.2e}")
            else:
                check(f"triangles, {stf} {peak:g}, {mode}: the slips in all", error, 1e-12)


def rectangles(check, rng, scratch):
    """
    The slips in all on rectangles, each case in a process of its own
    """
    import h5py
    Np, nt = 6, 80
    with h5py.File(scratch / "kinematicG.gf.h5", "w") as h5:
        h5["kinematicG.gf"] = numpy.zeros((2 * Np * nt, 10))
    with h5py.File(scratch / "kinematicG.data.h5", "w") as h5:
        h5["kinematicG.data"] = numpy.zeros(10)
    slip = numpy.stack([rng.normal(0, 0.3, (SAMPLES, Np)), rng.uniform(1, 3, (SAMPLES, Np))], axis=-1)
    θ = numpy.column_stack([slip[..., 0], slip[..., 1], rng.uniform(0.1, 5, (SAMPLES, Np)) * DT,
                            rng.uniform(2.0, 3.5, (SAMPLES, Np)), rng.uniform(5, 25, SAMPLES),
                            rng.uniform(5, 15, SAMPLES)])
    numpy.save(scratch / "theta.npy", θ)
    for stf, peak in [("triangle", 0.25), ("cosine", 0.5)]:
        for mode in ("sample", "step", "bin"):
            folder = scratch / f"{stf}-{mode}"
            folder.mkdir()
            (folder / "slipmodel.pfg").write_text(RECTANGLE.format(nt=nt, dt=DT, stf=stf, peak=peak,
                                                                   integration=mode, samples=SAMPLES))
            done = subprocess.run([sys.executable, __file__, "--rectangle", str(folder)], capture_output=True,
                                  text=True)
            if done.returncode:
                print(done.stdout[-2000:], done.stderr[-2000:])
                check(f"rectangles, {stf} {peak:g}, {mode}: ran", 1, 0)
                continue
            total = numpy.load(folder / "total.npy")
            error = numpy.abs(total - numpy.moveaxis(slip, -1, 1)).max() / numpy.abs(slip).max()
            if mode == "sample":
                print(f"  rectangles, {stf} {peak:g}, sample: the slips in all off by {error:.2e}")
            else:
                check(f"rectangles, {stf} {peak:g}, {mode}: the slips in all", error, 1e-12)


def rectangle(folder):
    """
    The slips in all of each patch on rectangles, for the samples in theta.npy
    """
    os.chdir(folder)
    sys.argv = [sys.argv[0]]
    import altar
    import altar.models.seismic
    app = altar.shells.altar(name="slipmodel").initialize()
    θ = numpy.load(folder.parent / "theta.npy")
    model = app.model
    Mb = numpy.zeros((θ.shape[0], model.NGbparameters))
    model.cmodel.cast_mb(theta=θ, mb=Mb, batch=θ.shape[0])
    numpy.save(folder / "total.npy", Mb.reshape(θ.shape[0], model.Nt, 2, -1).sum(axis=1))
    return 0


def main():
    # the reference asks quad for more than it can promise, harmlessly
    warnings.simplefilter("ignore", IntegrationWarning)
    rng = numpy.random.default_rng(7)
    failures = 0

    def check(label, error, tolerance):
        nonlocal failures
        ok = error <= tolerance
        failures += not ok
        print(f"  {label}: {error:.2e} (tolerance {tolerance:.0e}) {'ok' if ok else 'FAILED'}")

    triangles(check, rng)
    rectangles(check, rng, pathlib.Path(tempfile.mkdtemp(prefix="altar-integration-")))
    print("source time functions over the intervals:", "passed" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--rectangle" in sys.argv:
        sys.exit(rectangle(pathlib.Path(sys.argv[sys.argv.index("--rectangle") + 1])))
    sys.exit(main())


# end of file
