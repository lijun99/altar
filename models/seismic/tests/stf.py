#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the source time functions of the kinematic model on triangles, a triangle peaking early and a
# raised cosine, on a small planar fault with random green's functions: the gradient against
# central differences of the likelihood, on the cpu, and with --gpu also on cuda, whose slip
# histories must agree with the cpu's; each backend runs in a process of its own
import os
import pathlib
import subprocess
import sys
import tempfile
import numpy

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trimesh"))

OBS, NT, SIGMA, CHAINS = 120, 40, 1e-2, 6
CASES = [("triangle", 0.25), ("cosine", 0.5)]
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
            cd_std = {sigma}
        Nt = {nt}
        dt = 1.0
        stf = {stf}
        peak = {peak}
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
                prior = gaussian
                prior.mean = 2
            risetime:
                count = {patches}
                prior = gaussian
                prior.mean = 5
            rupturevelocity:
                count = {patches}
                prior = gaussian
                prior.mean = 3
            hypocenter:
                count = 2
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


def worker(scratch, stf, peak, gpu):
    """
    The slip histories, the gradient and its central differences of one backend, into a file
    """
    os.chdir(scratch)
    from kinematic_mesh import fault
    Np = len(fault("planar")[2])
    config = f"{stf}-{peak}-{'gpu' if gpu else 'cpu'}.pfg"
    (scratch / config).write_text(CONFIG.format(patches=Np, obs=OBS, sigma=SIGMA, nt=NT, stf=stf, peak=peak,
                                                gpus=int(gpu), chains=CHAINS))
    sys.argv = [sys.argv[0], f"--config={config}"]
    import altar
    import altar.models.seismic
    app = altar.shells.altar(name="slipmodel").initialize()
    model = app.model
    θ = numpy.load(scratch / "theta.npy")
    out = model.forward_problem(application=app, theta=θ)
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
    fd = numpy.column_stack([(loglike(θ + e) - loglike(θ - e)) / (2 * h) for e in numpy.identity(θ.shape[1]) * h])
    numpy.savez(scratch / f"{stf}-{peak}-{'gpu' if gpu else 'cpu'}.npz", Mb=out["Mb"], gradient=gradient, fd=fd)


def main():
    gpu = "--gpu" in sys.argv
    import h5py
    from kinematic_mesh import fault
    rng = numpy.random.default_rng(4)
    xyz, uv, faces, sources, owners = fault("planar")
    Np = len(faces)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-stf-"))
    with h5py.File(scratch / "mesh.h5", "w") as h5:
        h5["vertices"], h5["uv"], h5["faces"] = xyz, uv, faces
        h5["sources/uv"], h5["sources/patch"] = sources, owners
    with h5py.File(scratch / "kinematicG.gf.h5", "w") as h5:
        h5["kinematicG.gf"] = rng.standard_normal((2 * Np * NT, OBS)) * 1e-2
    with h5py.File(scratch / "kinematicG.data.h5", "w") as h5:
        h5["kinematicG.data"] = rng.standard_normal(OBS) * 1e-2
    θ = numpy.column_stack([rng.normal(0, 0.3, (CHAINS, Np)), rng.uniform(1, 3, (CHAINS, Np)),
                            rng.uniform(4, 6, (CHAINS, Np)), rng.uniform(2.5, 3.5, (CHAINS, Np)),
                            rng.uniform(12, 18, (CHAINS, 2))])
    numpy.save(scratch / "theta.npy", θ)

    failures = 0

    def check(label, error, tolerance):
        nonlocal failures
        ok = error <= tolerance
        failures += not ok
        print(f"  {label}: {error:.2e} (tolerance {tolerance:.0e}) {'ok' if ok else 'FAILED'}")

    for stf, peak in CASES:
        results = {}
        for backend in ["cpu"] + (["gpu"] if gpu else []):
            command = [sys.executable, __file__, "--worker", str(scratch), stf, str(peak)] + (["--gpu"] if backend == "gpu" else [])
            done = subprocess.run(command, capture_output=True, text=True)
            if done.returncode:
                print(done.stdout[-2000:], done.stderr[-2000:])
                failures += 1
                continue
            results[backend] = numpy.load(scratch / f"{stf}-{peak}-{backend}.npz")
            r = results[backend]
            check(f"{stf} {peak:g}, {backend}: gradient", numpy.abs(r["gradient"] - r["fd"]).max() / numpy.abs(r["fd"]).max(), 1e-6)
        if len(results) == 2:
            check(f"{stf} {peak:g}: slips, cuda against cpu",
                  numpy.abs(results["gpu"]["Mb"] - results["cpu"]["Mb"]).max() / numpy.abs(results["cpu"]["Mb"]).max(), 1e-12)

    print("source time functions:", "passed" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        worker(pathlib.Path(sys.argv[i + 1]), sys.argv[i + 2], float(sys.argv[i + 3]), "--gpu" in sys.argv)
        sys.exit(0)
    sys.exit(main())


# end of file
