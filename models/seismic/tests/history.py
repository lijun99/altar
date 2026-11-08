#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the slip histories of the kinematic model on triangles outside of an application, SlipHistory,
# against those of the model in one, on a small planar fault, for a batch of samples and for one
import os
import pathlib
import sys
import tempfile
import numpy

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trimesh"))

OBS, NT, DT, CHAINS, STF, PEAK = 20, 40, 1.0, 4, "triangle", 0.25
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
            observations = {obs}
            data_file = kinematicG.data.h5
            cd_std = 1
        Nt = {nt}
        dt = {dt}
        stf = {stf}
        peak = {peak}
        psets_list = [strikeslip, dipslip, risetime, rupturevelocity, hypocenter]
        psets:
            strikeslip = contiguous
            dipslip = contiguous
            risetime = contiguous
            rupturevelocity = contiguous
            hypocenter = contiguous
            strikeslip.count = {patches}
            dipslip.count = {patches}
            risetime.count = {patches}
            rupturevelocity.count = {patches}
            hypocenter.count = 2
    job:
        tasks = 1
        gpus = 0
        chains = {chains}
        precision = float64
"""


def main():
    import h5py
    from kinematic_mesh import fault
    rng = numpy.random.default_rng(5)
    xyz, uv, faces, sources, owners = fault("planar")
    Np = len(faces)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-history-"))
    with h5py.File(scratch / "mesh.h5", "w") as h5:
        h5["vertices"], h5["uv"], h5["faces"] = xyz, uv, faces
        h5["sources/uv"], h5["sources/patch"] = sources, owners
    with h5py.File(scratch / "kinematicG.gf.h5", "w") as h5:
        h5["kinematicG.gf"] = numpy.zeros((2 * Np * NT, OBS))
    with h5py.File(scratch / "kinematicG.data.h5", "w") as h5:
        h5["kinematicG.data"] = numpy.zeros(OBS)
    (scratch / "history.pfg").write_text(CONFIG.format(patches=Np, obs=OBS, nt=NT, dt=DT, stf=STF,
                                                        peak=PEAK, chains=CHAINS))

    # a batch of samples
    slip = numpy.stack([rng.normal(0, 0.3, (CHAINS, Np)), rng.uniform(1, 3, (CHAINS, Np))], axis=-1)
    risetime = rng.uniform(4, 6, (CHAINS, Np))
    velocity = rng.uniform(2.0, 3.5, (CHAINS, Np))
    hypocenter = rng.uniform(12, 18, (CHAINS, 2))
    θ = numpy.column_stack([slip[..., 0], slip[..., 1], risetime, velocity, hypocenter])

    # the model in an application
    os.chdir(scratch)
    sys.argv = [sys.argv[0], "--config=history.pfg"]
    import altar
    import altar.models.seismic
    app = altar.shells.altar(name="slipmodel").initialize()
    model = app.model
    expected = model.forward_problem(application=app, theta=θ)["Mb"].reshape(CHAINS, NT, 2, Np)
    times = numpy.asarray(model.cmodel.arrivals(theta=θ, batch=CHAINS)).reshape(CHAINS, -1)

    # and outside of one
    from altar.models.seismic.history import SlipHistory
    history = SlipHistory(xyz, faces, uv, sources, owners, Nt=NT, dt=DT, Nmesh=6, stf=STF, peak=PEAK)
    Mb = history(slip, risetime, velocity, hypocenter)
    one = history(slip[1], risetime[1], velocity[1], hypocenter[1])
    at_sources, at_vertices = history.arrivals(velocity, hypocenter)
    first, _ = history.arrivals(velocity[0], hypocenter[0])

    failures = 0

    def check(label, ok):
        nonlocal failures
        failures += not ok
        print(f"  {label}: {'ok' if ok else 'FAILED'}")

    scale = numpy.abs(expected).max()
    check(f"Mb of a batch, {Mb.shape}", Mb.shape == expected.shape and numpy.abs(Mb - expected).max() <= 1e-12 * scale)
    check(f"Mb of one sample, {one.shape}", one.shape == expected[1].shape and numpy.abs(one - expected[1]).max() <= 1e-12 * scale)
    check(f"rupture times at the vertices, {at_vertices.shape}", numpy.abs(at_vertices - times).max() <= 1e-12 * times.max())
    check(f"rupture times at the source points, {at_sources.shape}, {first.shape}",
          at_sources.shape == (CHAINS, len(sources)) and numpy.allclose(first, at_sources[0], rtol=0, atol=1e-12)
          and at_sources.min() >= 0)

    print("slip history:", "passed" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


# end of file
