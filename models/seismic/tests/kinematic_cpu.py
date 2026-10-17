#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# check the cpu kinematic models, the rectangular patches by fast sweeping and the triangulated
# faults (planar, a splay ending on a main fault, T, and crossing it, X) by the fast iterative
# method: the gradient of r.Mb against central differences; single precision against double; the
# meshes against the numpy prototype in trimesh/; the model's own likelihood gradient against
# central differences of its likelihood, the covariance merged into the data and not; and, with
# --gpu, the cpu models against the cuda ones. Makes its own faults and green's functions:
#     python kinematic_cpu.py [--gpu]
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import numpy

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trimesh"))
import kinematic_mesh as km

# the rectangular fault: 3 x 2 patches of 10 km, 6 x 6 cells each
NAS, NDD, NMESH, DSP, NPT, NT, DT, OBS = 3, 2, 6, 10.0, 2, 30, 1.0, 200
RECTANGLE = """
slipmodel:
    model = altar.models.seismic.kinematic
    model:
        case = .
        Nas = {nas}
        Ndd = {ndd}
        Nmesh = {nmesh}
        dsp = {dsp}
        Nt = {nt}
        Npt = {npt}
        dt = {dt}
        green = kinematicG.gf.h5
        dataobs:
            observations = {obs}
            data_file = kinematicG.data.h5
            cd_std = 1.0e-2
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
                prior.mean = 10
                prior.sigma = 5
    job.tasks = 1
    job.gpus = 1
    job.chains = 8
"""
PROBLEMS = ["rectangle", "planar", "T", "X"]


def setup(scratch, problem, rng):
    """
    The inputs of {problem} in {scratch}, its configuration, and random models
    """
    import h5py
    folder = scratch / problem
    folder.mkdir()
    if problem == "rectangle":
        Np = NAS * NDD
        config = RECTANGLE.format(nas=NAS, ndd=NDD, nmesh=NMESH, dsp=DSP, nt=NT, npt=NPT, dt=DT,
                                  obs=OBS, patches=Np)
        nt = NT
        hypocenters = numpy.column_stack([rng.uniform(2, 18, 8), rng.uniform(2, 8, 8)])
    else:
        xyz, uv, faces, sources, owners = km.fault(problem)
        Np = len(faces)
        with h5py.File(folder / "mesh.h5", "w") as h5:
            h5["vertices"], h5["uv"], h5["faces"] = xyz, uv, faces
            h5["sources/uv"], h5["sources/patch"] = sources, owners
        config = km.CONFIG.format(refine=km.REFINE, patches=Np, obs=km.OBS, nt=km.NT, dt=km.DT)
        nt = km.NT
        hypocenters = km.hypocenters(rng, problem, 8)
    obs = OBS if problem == "rectangle" else km.OBS
    with h5py.File(folder / "kinematicG.gf.h5", "w") as h5:
        h5["kinematicG.gf"] = rng.standard_normal((2 * Np * nt, obs)) * 1e-2
    with h5py.File(folder / "kinematicG.data.h5", "w") as h5:
        h5["kinematicG.data"] = rng.standard_normal(obs) * 1e-2
    (folder / "slipmodel.pfg").write_text(config)
    θ = numpy.column_stack([rng.normal(0, 0.5, (8, Np)), rng.uniform(0.5, 4, (8, Np)),
                            rng.uniform(3, 10, (8, Np)), rng.uniform(1.8, 3.5, (8, Np)), hypocenters])
    dMb = rng.standard_normal((8, 2 * Np * nt))
    numpy.savez(folder / "inputs.npz", theta=θ, dMb=dMb)
    return folder, Np


def main():
    gpu = "--gpu" in sys.argv
    rng = numpy.random.default_rng(5)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-kinematic-cpu-"))
    failures = 0
    try:
        for problem in PROBLEMS:
            folder, Np = setup(scratch, problem, rng)
            results = {}
            runs = [("cpu", "float64"), ("cpu", "float32"), ("cpu", "merged")]
            runs += [("cuda", "float64")] if gpu else []
            for backend, precision in runs:
                label = f"{backend}-{precision}"
                status = subprocess.run([sys.executable, str(HERE / "kinematic_cpu.py"), "--run", backend,
                                         precision, problem], cwd=folder, capture_output=True, text=True)
                if status.returncode != 0:
                    print(status.stdout + status.stderr)
                    return 1
                print("\n".join(l for l in status.stdout.splitlines() if l.startswith(problem)))
                results[label] = numpy.load(folder / f"{label}.npz", allow_pickle=True)
            failures += compare(problem, folder, Np, results, gpu)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 1 if failures else 0


def compare(problem, folder, Np, results, gpu):
    """
    The checks of {problem}; the number that failed
    """
    rel = lambda a, b: numpy.abs(a - b).max() / numpy.abs(b).max()
    c64, c32 = results["cpu-float64"], results["cpu-float32"]
    checks = [
        ("gradient against central differences", float(c64["fd"]), 1e-6),
        ("likelihood gradient against central differences", float(c64["e2e"]), 1e-6),
        ("the same, covariance merged into the data", float(results["cpu-merged"]["e2e"]), 1e-6),
        ("Mb, single against double precision", rel(c32["Mb"], c64["Mb"]), 1e-5),
        ("gradient, single against double precision", rel(c32["grad"], c64["grad"]), 1e-3),
    ]
    if problem != "rectangle":
        # the prototype, on the same mesh
        from altar.models.seismic import mesh as meshes
        from mesh import Mesh
        from slip import SlipHistory
        xyz, uv, faces, sources, owners = km.fault(problem)
        points, coords, triangles, patch = meshes.refine(xyz, faces, uv, km.REFINE)
        proto = SlipHistory(Mesh(points, triangles, patch, coords), sources, owners, Np, km.NT, km.DT,
                            numpy.zeros(Np), len(owners) // Np)
        inputs = numpy.load(folder / "inputs.npz")
        mb, pg = [], []
        for t, d in zip(inputs["theta"], inputs["dMb"]):
            mb.append(proto(t))
            pg.append(proto.adjoint(d))
        checks += [("Mb against the prototype", rel(c64["Mb"], numpy.array(mb)), 1e-10),
                   ("gradient against the prototype", rel(c64["grad"], numpy.array(pg)), 1e-9)]
    if gpu:
        g64 = results["cuda-float64"]
        checks += [("Mb, cpu against cuda", rel(c64["Mb"], g64["Mb"]), 1e-10),
                   ("gradient, cpu against cuda", rel(c64["grad"], g64["grad"]), 1e-9)]
    failures = 0
    for name, value, tolerance in checks:
        good = value <= tolerance
        failures += not good
        print(f"{problem:9s} {name:50s} {'ok' if good else 'FAIL':4s} {value:.2e} (tolerance {tolerance:.0e})")
    return failures


def run(backend, precision, problem):
    """
    Mb, the gradient of r.Mb for the saved r, and, on the cpu in double precision, its worst
    relative difference from central differences and the likelihood check; "merged", that check
    with the covariance merged into the data
    """
    gpus = 1 if backend == "cuda" else 0
    label = f"{backend}-{precision}"
    merged = precision == "merged"
    precision = "float64" if merged else precision
    sys.argv = [sys.argv[0], "--config=slipmodel.pfg", f"--job.precision={precision}", f"--job.gpus={gpus}"]
    if merged:
        sys.argv.append("--model.dataobs.merge_cd_with_data=True")
    import altar, altar.models.seismic
    app = altar.shells.altar(name="slipmodel").initialize()
    model = app.model
    inputs = numpy.load("inputs.npz")
    θ, dMb = inputs["theta"], inputs["dMb"]
    count = θ.shape[0]
    fd = e2e = None
    if backend == "cuda":
        import altar.cuda
        gθ = altar.cuda.matrix(source=θ, dtype=precision)
        gMb = altar.cuda.matrix(shape=(count, model.NGbparameters), dtype=precision)
        gdMb = altar.cuda.matrix(source=dMb, dtype=precision)
        gGrad = altar.cuda.matrix(shape=θ.shape, dtype=precision)
        model.cmodel.cast_mb(theta=gθ.grid, mb=gMb.grid, batch=count)
        model.cmodel.gradient(theta=gθ.grid, dmb=gdMb.grid, gradient=gGrad.grid, batch=count)
        Mb, grad = (numpy.array(numpy.asarray(g), dtype=float) for g in (gMb, gGrad))
    else:
        t = numpy.ascontiguousarray(θ, dtype=precision)
        d = numpy.ascontiguousarray(dMb, dtype=precision)
        Mb = numpy.zeros((count, model.NGbparameters), dtype=precision)
        grad = numpy.zeros_like(t)
        model.cmodel.cast_mb(theta=t, mb=Mb, batch=count)
        model.cmodel.gradient(theta=t, dmb=d, gradient=grad, batch=count)
        Mb, grad = Mb.astype(float), grad.astype(float)
        if merged:
            e2e = likelihood_gradient(model, θ)
        elif precision == "float64":
            fd = differences(model, θ, dMb, grad)
            e2e = likelihood_gradient(model, θ)
    print(f"{problem}: {label}, {count} samples, {model.NGbparameters} slip windows")
    numpy.savez(f"{label}.npz", Mb=Mb, grad=grad, fd=numpy.array(fd, dtype=object),
                e2e=numpy.array(e2e, dtype=object))
    return 0


def differences(model, θ, dMb, grad):
    """
    The worst relative difference of {grad} from central differences of r.Mb, all the parameters of
    all the samples at once
    """
    count = θ.shape[0]
    Mb = numpy.zeros((count, model.NGbparameters))
    def f(x):
        model.cmodel.cast_mb(theta=numpy.ascontiguousarray(x), mb=Mb, batch=count)
        return (Mb * dMb).sum(1)
    ε = 1e-6
    fds = numpy.zeros_like(θ)
    for k in range(θ.shape[1]):
        e = numpy.zeros_like(θ)
        e[:, k] = ε
        fds[:, k] = (f(θ + e) - f(θ - e)) / (2 * ε)
    return (numpy.abs(grad - fds).max(axis=1) / numpy.abs(fds).max(axis=1)).max()


def likelihood_gradient(model, θ):
    """
    The model's own data likelihood gradient against central differences of its data likelihood,
    along random directions: the worst relative difference
    """
    import types
    count = θ.shape[0]
    model.checked_unbounded_priors = True
    step = types.SimpleNamespace(theta=θ.copy(), grad_prior=numpy.zeros_like(θ), grad_data=numpy.zeros_like(θ))
    model.gradient(controller=None, step=step, batch=count)
    def L(x):
        likelihood = numpy.zeros(count)
        model.eval_data_likelihood(theta=x, likelihood=likelihood, batch=count)
        return likelihood
    rng = numpy.random.default_rng(1)
    worst = 0.0
    for trial in range(3):
        v = rng.standard_normal(θ.shape)
        ε = 1e-6
        fd = (L(θ + ε * v) - L(θ - ε * v)) / (2 * ε)
        an = (step.grad_data * v).sum(1)
        worst = max(worst, (numpy.abs(an - fd) / numpy.abs(fd)).max())
    return worst


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        sys.exit(run(sys.argv[2], sys.argv[3], sys.argv[4]))
    sys.exit(main())


# end of file
