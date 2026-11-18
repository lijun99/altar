#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# a synthetic for kinematicgaussianslip: one front on the planar fault of the tests, slips drawn
# from their prior, binned data of 3 components x 15 stations x 12 bins with correlated noise;
# then, after a run of catmip, the posterior of the front and of the slips against the truth
#     python synthetic.py setup <folder>
#     cd <folder> && slipmodel --config=slipmodel.pfg
#     python synthetic.py evaluate <folder>
import os
import pathlib
import sys
import numpy

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "trimesh"))

COMPONENTS, STATIONS, BINS, DT, NMESH, SIGMA = 3, 15, 12, 2.0, 6, 0.7
# rise time, velocity (km/day), hypocenter (km), origin (day)
TRUTH = {"risetime": [3.0], "rupturevelocity": [5.0], "hypocenter": [10.0, 15.0], "origin": [1.0]}
CONFIG = """
slipmodel:
    model = altar.models.seismic.kinematicgaussianslip
    model:
        case = .
        mesh = mesh.h5
        Nmesh = {nmesh}
        green = static.gf.h5
        components = {components}
        bins = {bins}
        dt = {dt}
        slip_sigma = {sigma}
        dataobs = altar.data.datakron
        dataobs:
            observations = {obs}
            data_file = data.h5
            noise_file = noise.h5
        psets_list = [risetime, rupturevelocity, hypocenter, origin]
        psets:
            risetime = contiguous
            rupturevelocity = contiguous
            hypocenter = contiguous
            origin = contiguous
            risetime:
                count = 1
                prior = uniform
                prior.support = (0.5, 10)
            rupturevelocity:
                count = 1
                prior = uniform
                prior.support = (1, 15)
            hypocenter:
                count = 2
                prior = uniform
                prior.support = (0, 40)
            origin:
                count = 1
                prior = uniform
                prior.support = (-5, 10)
    controller = altar.bayesian.catmip
    controller:
        archiver = altar.bayesian.h5recorder
        archiver:
            output_dir = results
    job:
        tasks = 1
        gpus = 0
        chains = 1024
        steps = 30
        precision = float64
"""


def setup(folder):
    """
    The inputs of the synthetic, the true slips and the configuration, in {folder}
    """
    import h5py
    from kinematic_mesh import fault
    import datakron
    folder.mkdir(parents=True, exist_ok=True)
    rng = numpy.random.default_rng(2026)
    xyz, uv, faces, sources, owners = fault("planar")
    Np = len(faces)
    with h5py.File(folder / "mesh.h5", "w") as h5:
        h5["vertices"], h5["uv"], h5["faces"] = xyz, uv, faces
        h5["sources/uv"], h5["sources/patch"] = sources, owners
    datakron.STATIONS, datakron.BINS = STATIONS, BINS
    S, R = datakron.noise(rng)
    with h5py.File(folder / "noise.h5", "w") as h5:
        h5["S"], h5["R"] = S, R
    G = rng.standard_normal((2 * Np, COMPONENTS * STATIONS))
    with h5py.File(folder / "static.gf.h5", "w") as h5:
        h5["static.gf"] = G
    (folder / "slipmodel.pfg").write_text(CONFIG.format(
        nmesh=NMESH, components=COMPONENTS, bins=BINS, dt=DT, sigma=SIGMA,
        obs=COMPONENTS * STATIONS * BINS))
    # the true front's slip fractions, from a model of the configuration, on placeholder data
    with h5py.File(folder / "data.h5", "w") as h5:
        h5["data"] = numpy.zeros(COMPONENTS * STATIONS * BINS)
    θ = numpy.concatenate([TRUTH[k] for k in ["risetime", "rupturevelocity", "hypocenter", "origin"]])
    # altar reads the command line on its first import, here
    model = application(folder).model
    from altar.data.Kronecker import Kronecker
    noise = Kronecker(S, R)
    F = model.fractions(θ[None])
    slip = rng.normal(0, SIGMA, 2 * Np)
    signal = model.marginal.predict(F, slip[None])[0]
    d = signal + noise.draw(rng)
    with h5py.File(folder / "data.h5", "w") as h5:
        h5["data"] = d
    numpy.savetxt(folder / "truth.txt", θ)
    numpy.savetxt(folder / "slip.txt", slip)
    print(f"setup: {Np} patches, {d.size} data, signal rms {numpy.sqrt((signal ** 2).mean()):.2f}, "
          f"noise rms {numpy.sqrt(noise.variance().mean()):.2f}")
    return 0


def application(folder):
    """
    The application of the configuration in {folder}
    """
    os.chdir(folder)
    sys.argv = [sys.argv[0], "--config=slipmodel.pfg"]
    import altar
    import altar.models.seismic
    return altar.shells.altar(name="slipmodel").initialize()


def evaluate(folder):
    """
    The posterior of the front and of the slips against the truth
    """
    model = application(folder).model
    from altar.actions.Forward import load_samples
    truth = numpy.loadtxt(folder / "truth.txt")
    slip = numpy.loadtxt(folder / "slip.txt")
    θ = load_samples(model=model, path=folder / "results" / "step_final.h5")
    mean, sd = θ.mean(axis=0), θ.std(axis=0)
    labels = ["rise time", "velocity", "hypocenter u", "hypocenter v", "origin"]
    print(f"evaluate: {θ.shape[0]} samples")
    for label, t, m, s in zip(labels, truth, mean, sd):
        print(f"  {label:14s} truth {t:7.3f}  posterior {m:7.3f} ± {s:6.3f}  z {(m - t) / s:6.2f}")
    # the slips: a draw for each of 256 samples
    rng = numpy.random.default_rng(1)
    rows = θ[rng.choice(θ.shape[0], min(256, θ.shape[0]), replace=False)]
    draws = model.draw_slips(rows, rng)[:, 0, :]
    m, s = draws.mean(axis=0), draws.std(axis=0)
    z = (slip - m) / s
    inside = (numpy.abs(z) <= 1).mean(), (numpy.abs(z) <= 2).mean()
    print(f"  slips: correlation of the mean with the truth {numpy.corrcoef(m, slip)[0, 1]:.3f}, rms z "
          f"{numpy.sqrt((z ** 2).mean()):.2f}, within 1 sd {inside[0]:.0%}, within 2 sd {inside[1]:.0%}")
    return 0


if __name__ == "__main__":
    command, folder = sys.argv[1], pathlib.Path(sys.argv[2]).resolve()
    sys.exit({"setup": setup, "evaluate": evaluate}[command](folder))


# end of file
