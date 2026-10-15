#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# check the cuda kinematic model on a triangulated fault against the numpy prototype in trimesh/:
# the slips over time Mb, and the gradient of a function of Mb, which must agree to round-off in
# double precision; the gradient against central differences of the cuda forward model; and
# single precision against double. Needs a gpu; makes its own small fault and green's functions:
#     python kinematic_mesh.py
import os
import pathlib
import shutil
import sys
import tempfile
import numpy

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "trimesh"))

# the fault: 4 x 3 squares of 10 km, each cut into two triangles, dipping 30 degrees
NS, ND, SIZE, DIP = 4, 3, 10.0, 30.0
REFINE, POINTS, NT, DT, OBS = 6, 2, 40, 1.0, 300
# the patches that share a velocity, in the grouped case
GROUP = 4
CONFIG = """
slipmodel:
    model = altar.models.seismic.kinematicmesh
    model:
        case = .
        mesh = mesh.h5
        refine = {refine}
        patches = {patches}
        green = kinematicG.gf.h5
        dataobs:
            observations = {obs}
            data_file = kinematicG.data.h5
            cd_std = 1.0e-2
        Nt = {nt}
        dt = {dt}
        psets_list = [strikeslip, dipslip, risetime, rupturevelocity, hypocenter]
        psets:
            strikeslip = contiguous
            dipslip = contiguous
            risetime = contiguous
            rupturevelocity = contiguous
            hypocenter = contiguous
            strikeslip:
                count = {patches}
                prior = gaussian
                prior.sigma = 0.5
            dipslip:
                count = {patches}
                prior = uniform
                prior.support = (0, 5)
            risetime:
                count = {patches}
                prior = uniform
                prior.support = (2, 12)
            rupturevelocity:
                count = {patches}
                prior = uniform
                prior.support = (1.5, 4)
            hypocenter:
                count = 2
                prior = gaussian
                prior.mean = 15
                prior.sigma = 5
    job.tasks = 1
    job.gpus = 1
    job.chains = 8
"""


def fault():
    """
    The vertices, surface coordinates and faces of the fault, and the source points of each patch,
    as csi's TriangularPatchesKin saves them
    """
    u = numpy.arange(NS + 1) * SIZE
    v = numpy.arange(ND + 1) * SIZE
    U, V = numpy.meshgrid(u, v, indexing="ij")
    uv = numpy.column_stack([U.ravel(), V.ravel()])
    δ = numpy.radians(DIP)
    xyz = numpy.column_stack([uv[:, 1] * numpy.cos(δ), uv[:, 0], 5.0 + uv[:, 1] * numpy.sin(δ)])
    index = lambda i, j: i * (ND + 1) + j
    faces = []
    for i in range(NS):
        for j in range(ND):
            a, b, c, d = index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)
            faces += [(a, b, c), (a, c, d)] if (i + j) % 2 else [(a, b, d), (b, c, d)]
    faces = numpy.array(faces)
    # the centroids of the n x n sub-triangles of each face
    n = POINTS
    bary = []
    for i in range(n):
        for j in range(n - i):
            bary.append(((i + 1 / 3) / n, (j + 1 / 3) / n))
            if i + j < n - 1:
                bary.append(((i + 2 / 3) / n, (j + 2 / 3) / n))
    w = numpy.column_stack([1 - numpy.sum(bary, 1), bary])
    sources = numpy.concatenate([w @ uv[f] for f in faces])
    owners = numpy.repeat(numpy.arange(len(faces)), len(w))
    return xyz, uv, faces, sources, owners


def theta(rng, Np, count, velocities=None):
    """
    {count} random models, with a velocity per patch or per group
    """
    velocities = Np if velocities is None else velocities
    return numpy.column_stack([rng.normal(0, 0.5, (count, Np)), rng.uniform(0.5, 4, (count, Np)),
                               rng.uniform(3, 10, (count, Np)), rng.uniform(1.8, 3.5, (count, velocities)),
                               rng.uniform(5, 30, (count, 1)), rng.uniform(5, 25, (count, 1))])


def main():
    import h5py
    rng = numpy.random.default_rng(7)
    xyz, uv, faces, sources, owners = fault()
    Np = len(faces)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-kinematic-mesh-"))
    failures = 0
    try:
        with h5py.File(scratch / "mesh.h5", "w") as h5:
            h5["vertices"], h5["uv"], h5["faces"] = xyz, uv, faces
            h5["sources/uv"], h5["sources/patch"] = sources, owners
            h5["groups/velocity"] = numpy.arange(Np) // GROUP
        with h5py.File(scratch / "kinematicG.gf.h5", "w") as h5:
            h5["kinematicG.gf"] = rng.standard_normal((2 * Np * NT, OBS)) * 1e-2
        with h5py.File(scratch / "kinematicG.data.h5", "w") as h5:
            h5["kinematicG.data"] = rng.standard_normal(OBS) * 1e-2
        config = CONFIG.format(refine=REFINE, patches=Np, obs=OBS, nt=NT, dt=DT)
        (scratch / "slipmodel.pfg").write_text(config)
        # the same, with a velocity for each group of {GROUP} patches
        grouped = config.replace(f"        refine = {REFINE}\n",
                                 f"        refine = {REFINE}\n        velocity_groups = groups/velocity\n")
        grouped = grouped.replace("            rupturevelocity:\n                count = " + str(Np),
                                  "            rupturevelocity:\n                count = " + str(Np // GROUP))
        (scratch / "grouped.pfg").write_text(grouped)
        os.chdir(scratch)

        # each precision in its own process, since pyre reads the command line once
        import subprocess
        results = {}
        for label in ["float64", "float32", "grouped"]:
            status = subprocess.run([sys.executable, str(HERE / "kinematic_mesh.py"), "--run", label,
                                     str(Np)], cwd=scratch, capture_output=True, text=True)
            lines = [l for l in (status.stdout + status.stderr).splitlines() if l.startswith(label)]
            print("\n".join(lines) if status.returncode == 0 else status.stdout + status.stderr)
            if status.returncode != 0:
                return 1
            saved = numpy.load(scratch / f"{label}.npz", allow_pickle=True)
            results[label] = (saved["theta"], saved["Mb"], saved["dMb"], saved["grad"],
                              saved["fd"].item())
        # the prototype, on the same mesh
        from altar.models.seismic import mesh as meshes
        from mesh import Mesh
        from slip import SlipHistory
        points, coords, triangles, patch = meshes.refine(xyz, faces, uv, REFINE)
        proto = SlipHistory(Mesh(points, triangles, patch, coords), sources, owners, Np, NT, DT,
                            numpy.zeros(Np), len(owners) // Np)
        θ, Mb, dMb, grad, fd = results["float64"]
        mb = numpy.array([proto(t) for t in θ])
        pg = []
        for t, d in zip(θ, dMb):
            proto(t)
            pg.append(proto.adjoint(d))
        pg = numpy.array(pg)
        checks = [
            ("Mb, cuda against the prototype", numpy.abs(Mb - mb).max() / numpy.abs(mb).max(), 1e-10),
            ("gradient, cuda against the prototype", numpy.abs(grad - pg).max() / numpy.abs(pg).max(), 1e-9),
            ("gradient, cuda against central differences", fd, 1e-6),
        ]
        θ32, Mb32, _, grad32, _ = results["float32"]
        checks += [
            ("Mb, single against double precision", numpy.abs(Mb32 - Mb).max() / numpy.abs(Mb).max(), 1e-5),
            ("gradient, single against double precision",
             numpy.abs(grad32 - grad).max() / numpy.abs(grad).max(), 1e-3),
        ]
        # the grouped velocities: the prototype on theta with each patch's velocity its group's, the
        # velocity gradients summed over each group
        θg, Mbg, dMbg, gradg, fdg = results["grouped"]
        groups = numpy.arange(Np) // GROUP
        G = Np // GROUP
        expand = lambda t: numpy.concatenate([t[:3 * Np], t[3 * Np:3 * Np + G][groups], t[3 * Np + G:]])
        mbg, pgg = [], []
        for t, d in zip(θg, dMbg):
            mbg.append(proto(expand(t)))
            g = proto.adjoint(d)
            pgg.append(numpy.concatenate([g[:3 * Np], numpy.bincount(groups, weights=g[3 * Np:4 * Np]),
                                          g[4 * Np:]]))
        mbg, pgg = numpy.array(mbg), numpy.array(pgg)
        checks += [
            ("grouped velocities: Mb against the prototype",
             numpy.abs(Mbg - mbg).max() / numpy.abs(mbg).max(), 1e-10),
            ("grouped velocities: gradient against the prototype",
             numpy.abs(gradg - pgg).max() / numpy.abs(pgg).max(), 1e-9),
            ("grouped velocities: against central differences", fdg, 1e-6),
        ]
        for name, value, tolerance in checks:
            good = value <= tolerance
            failures += not good
            print(f"{name:52s} {'ok' if good else 'FAIL':4s} {value:.2e} (tolerance {tolerance:.0e})")
    finally:
        os.chdir(HERE)
        shutil.rmtree(scratch, ignore_errors=True)
    return 1 if failures else 0


def run(label, Np, rng):
    """
    Mb, the gradient of r.Mb for random r, and its worst relative difference from central
    differences, for random models: in double or single precision, or in double with the
    velocities of groups of patches
    """
    precision = "float32" if label == "float32" else "float64"
    config = "grouped.pfg" if label == "grouped" else "slipmodel.pfg"
    # pyre reads the command line when altar is first imported
    sys.argv = [sys.argv[0], f"--config={config}", f"--job.precision={precision}"]
    import altar, altar.cuda, altar.models.seismic
    app = altar.shells.altar(name="slipmodel").initialize()
    model = app.model
    count = model.samples
    θ = theta(rng, Np, count, velocities=Np // GROUP if label == "grouped" else None)
    dMb = rng.standard_normal((count, model.NGbparameters))
    gθ = altar.cuda.matrix(source=θ, dtype=precision)
    gMb = altar.cuda.matrix(shape=(count, model.NGbparameters), dtype=precision)
    gdMb = altar.cuda.matrix(source=dMb, dtype=precision)
    gGrad = altar.cuda.matrix(shape=θ.shape, dtype=precision)
    model.cmodel.cast_mb(theta=gθ.grid, mb=gMb.grid, batch=count)
    Mb = numpy.array(numpy.asarray(gMb), dtype=float)
    model.cmodel.gradient(theta=gθ.grid, dmb=gdMb.grid, gradient=gGrad.grid, batch=count)
    grad = numpy.array(numpy.asarray(gGrad), dtype=float)
    fd = None
    if label != "float32":
        # central differences of r.Mb, all the parameters of all the samples at once
        def f(x):
            g = altar.cuda.matrix(source=x, dtype=precision)
            model.cmodel.cast_mb(theta=g.grid, mb=gMb.grid, batch=count)
            return (numpy.asarray(gMb) * dMb).sum(1)
        ε = 1e-6
        fds = numpy.zeros_like(θ)
        for k in range(θ.shape[1]):
            e = numpy.zeros_like(θ)
            e[:, k] = ε
            fds[:, k] = (f(θ + e) - f(θ - e)) / (2 * ε)
        fd = (numpy.abs(grad - fds).max(axis=1) / numpy.abs(fds).max(axis=1)).max()
    print(f"{label}: {count} samples, {model.vertices} vertices,"
          f" eikonal passes {model.cmodel.iterations(count)}")
    numpy.savez(f"{label}.npz", theta=θ, Mb=Mb, dMb=dMb, grad=grad, fd=numpy.array(fd, dtype=object))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        sys.exit(run(sys.argv[2], int(sys.argv[3]), rng=numpy.random.default_rng(11)))
    sys.exit(main())


# end of file
