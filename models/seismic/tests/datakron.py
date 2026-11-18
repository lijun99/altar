#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the data with a covariance separable in space and time, altar.data.datakron, on a small gnss-like
# problem of 3 components x 6 stations x 8 bins: the static model and the kinematic one on
# triangles, their data likelihoods and gradients against those with the same covariance dense,
# altar.data.datal2 with a cd_file, and the likelihoods against the dense gaussian; on the cpu,
# the covariance merged into the data and not, and with --gpu also on cuda; each run in a process
# of its own
import pathlib
import subprocess
import sys
import tempfile
import numpy

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trimesh"))

COMPONENTS, STATIONS, BINS, CHAINS, PATCHES, NT = 3, 6, 8, 5, 5, 20
OBS = COMPONENTS * STATIONS * BINS
STATIC = """
slipmodel:
    model = altar.models.seismic.static
    model:
        case = {case}
        patches = {patches}
        green = static.gf.h5
        dataobs = {data}
        dataobs:
            observations = {obs}
            data_file = data.h5
            {covariance}
            merge_cd_with_data = {merge}
        psets_list = [strikeslip, dipslip]
        psets:
            strikeslip = contiguous
            dipslip = contiguous
            strikeslip:
                count = {patches}
                prior = gaussian
            dipslip:
                count = {patches}
                prior = gaussian
    job:
        tasks = 1
        gpus = {gpus}
        chains = {chains}
        precision = float64
        gpuprecision = float64
"""
KINEMATIC = """
slipmodel:
    model = altar.models.seismic.kinematicmesh
    model:
        case = {case}
        mesh = mesh.h5
        Nmesh = 6
        patches = {patches}
        green = kinematicG.gf.h5
        Nt = {nt}
        dt = 1.0
        integration = bin
        dataobs = {data}
        dataobs:
            observations = {obs}
            data_file = data.h5
            {covariance}
            merge_cd_with_data = {merge}
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
            dipslip:
                count = {patches}
                prior = gaussian
            risetime:
                count = {patches}
                prior = gaussian
                prior.mean = 3
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


def noise(rng):
    """
    S and R as sse_toolkit fits them: scaled exponential correlation of the stations, and a
    common level, white noise, a random walk and an exponential term over the bins
    """
    xy = rng.uniform(0, 300, (STATIONS, 2))
    d = numpy.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    i = numpy.arange(BINS)
    S, R = [], []
    for c, (c0, L) in enumerate([(0.4, 150.0), (0.3, 200.0), (0.2, 100.0)]):
        sigma = rng.uniform(0.5, 2.0, STATIONS) * (3 if c == 2 else 1)
        rho = c0 * numpy.exp(-d / L)
        numpy.fill_diagonal(rho, 1.0)
        S.append(sigma[:, None] * sigma[None, :] * rho)
        a, b, w, e, tau = 0.3, 0.4, 0.01 * (c + 1), 0.2, 3.0
        R.append(a + b * numpy.eye(BINS) + w * (numpy.minimum.outer(i, i) + 1.0)
                 + e * numpy.exp(-numpy.abs(numpy.subtract.outer(i, i)) / tau))
    return numpy.array(S), numpy.array(R)


def worker(case, kind, data, backend, merge):
    """
    The data likelihoods and gradients of one run, into a file
    """
    gpu = backend == "cuda"
    covariance = "noise_file = noise.h5" if data == "datakron" else "cd_file = cd.h5"
    template = STATIC if kind == "static" else KINEMATIC
    name = f"{kind}-{data}-{backend}-{merge}"
    folder = case / name
    folder.mkdir()
    Np = PATCHES if kind == "static" else int(numpy.load(case / "patches.npy"))
    (folder / "slipmodel.pfg").write_text(template.format(
        case=case, patches=Np, data=f"altar.data.{data}", obs=OBS, covariance=covariance,
        merge=merge, nt=NT, gpus=int(gpu), chains=CHAINS))
    import os
    os.chdir(folder)
    sys.argv = [sys.argv[0]]
    import altar
    import altar.models.seismic
    app = altar.shells.altar(name="slipmodel").initialize()
    model = app.model
    θ = numpy.load(case / f"{kind}-theta.npy")
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
    likelihood = numpy.array(step.data)
    if gpu:
        step.prior_gradient = altar.cuda.matrix(shape=θ.shape, dtype="float64").zero()
        step.data_gradient = altar.cuda.matrix(shape=θ.shape, dtype="float64").zero()
        model.gradient(controller=None, step=step)
        gradient = numpy.array(step.data_gradient)
    else:
        step.theta, step.grad_prior, step.grad_data = θ.copy(), numpy.zeros_like(θ), numpy.zeros_like(θ)
        model.gradient(controller=None, step=step)
        gradient = step.grad_data
    # a raw residual, whitened by the data
    prediction = model.forward_problem(application=app, theta=θ)["data"]
    raw = array(numpy.zeros(CHAINS))
    model.dataobs.eval_likelihood(prediction=array(prediction - model.dataobs.observed()), likelihood=raw,
                                  residual=True, whitened=False)
    out = dict(likelihood=likelihood, gradient=gradient, prediction=prediction, raw=numpy.array(raw),
               sigma=model.dataobs.sigma())
    if kind == "static" and not gpu:
        mean, cstar, evidence = model.conjugate_posterior(mean=numpy.zeros(θ.shape[1]),
                                                          variance=numpy.ones(θ.shape[1]))
        out.update(mean=mean, cstar=cstar, evidence=evidence)
    numpy.savez(case / f"{name}.npz", **out)
    return 0


def main():
    gpu = "--gpu" in sys.argv
    import h5py
    from kinematic_mesh import fault
    rng = numpy.random.default_rng(11)
    case = pathlib.Path(tempfile.mkdtemp(prefix="altar-datakron-"))
    S, R = noise(rng)
    from altar.data.Kronecker import Kronecker
    kron = Kronecker(S, R)
    Cd = kron.dense()
    with h5py.File(case / "noise.h5", "w") as h5:
        h5["S"], h5["R"] = S, R
    with h5py.File(case / "cd.h5", "w") as h5:
        h5["cd"] = Cd
    # the static model
    G = rng.standard_normal((OBS, 2 * PATCHES))
    with h5py.File(case / "static.gf.h5", "w") as h5:
        h5["static.gf"] = G
    numpy.save(case / "static-theta.npy", rng.normal(0, 1, (CHAINS, 2 * PATCHES)))
    # the kinematic model on triangles
    xyz, uv, faces, sources, owners = fault("planar")
    Np = len(faces)
    numpy.save(case / "patches.npy", Np)
    with h5py.File(case / "mesh.h5", "w") as h5:
        h5["vertices"], h5["uv"], h5["faces"] = xyz, uv, faces
        h5["sources/uv"], h5["sources/patch"] = sources, owners
    with h5py.File(case / "kinematicG.gf.h5", "w") as h5:
        h5["kinematicG.gf"] = rng.standard_normal((2 * Np * NT, OBS))
    numpy.save(case / "kinematic-theta.npy", numpy.column_stack([
        rng.normal(0, 0.3, (CHAINS, Np)), rng.uniform(1, 3, (CHAINS, Np)), rng.uniform(1, 5, (CHAINS, Np)),
        rng.uniform(2.5, 3.5, (CHAINS, Np)), rng.uniform(12, 18, (CHAINS, 2))]))
    # data: a prediction of the static model plus correlated noise
    d = G @ rng.normal(0, 1, 2 * PATCHES) + kron.draw(rng)
    with h5py.File(case / "data.h5", "w") as h5:
        h5["data"] = d

    failures = 0

    def check(label, error, tolerance):
        nonlocal failures
        ok = error <= tolerance
        failures += not ok
        print(f"  {label}: {error:.2e} (tolerance {tolerance:.0e}) {'ok' if ok else 'FAILED'}")

    rel = lambda a, b: numpy.abs(a - b).max() / numpy.abs(b).max()
    Ci = numpy.linalg.inv(Cd)
    constant = -0.5 * (OBS * numpy.log(2 * numpy.pi) + numpy.linalg.slogdet(Cd)[1])
    runs = [("cpu", "False"), ("cpu", "True")] + ([("cuda", "True")] if gpu else [])
    for kind in ("static", "kinematic"):
        for backend, merge in runs:
            results = {}
            for data in ("datakron", "datal2"):
                command = [sys.executable, __file__, "--worker", str(case), kind, data, backend, merge]
                done = subprocess.run(command, capture_output=True, text=True)
                if done.returncode:
                    print(done.stdout[-3000:], done.stderr[-3000:])
                    failures += 1
                    continue
                results[data] = numpy.load(case / f"{kind}-{data}-{backend}-{merge}.npz")
            if len(results) < 2:
                continue
            kr, dense = results["datakron"], results["datal2"]
            label = f"{kind}, {backend}, merged {merge}"
            r = kr["prediction"] - d
            gaussian = constant - 0.5 * numpy.einsum("ki,ij,kj->k", r, Ci, r)
            check(f"{label}: likelihood against the dense gaussian", rel(kr["likelihood"], gaussian), 1e-11)
            check(f"{label}: likelihood against datal2", rel(kr["likelihood"], dense["likelihood"]), 1e-11)
            r = dense["prediction"] - d
            check(f"{label}: datal2 likelihood against the dense gaussian",
                  rel(dense["likelihood"], constant - 0.5 * numpy.einsum("ki,ij,kj->k", r, Ci, r)), 1e-11)
            check(f"{label}: likelihood of a raw residual against the dense gaussian", rel(kr["raw"], gaussian), 1e-11)
            check(f"{label}: gradient against datal2", rel(kr["gradient"], dense["gradient"]), 1e-10)
            check(f"{label}: sigma against datal2", rel(kr["sigma"], dense["sigma"]), 1e-14)
            if "mean" in kr:
                check(f"{label}: conjugate posterior mean against datal2", rel(kr["mean"], dense["mean"]), 1e-10)
                check(f"{label}: conjugate posterior covariance against datal2", rel(kr["cstar"], dense["cstar"]), 1e-10)
                check(f"{label}: evidence against datal2", abs(kr["evidence"] - dense["evidence"]) / abs(dense["evidence"]), 1e-11)
    print("datakron:", "passed" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        sys.exit(worker(pathlib.Path(sys.argv[i + 1]), *sys.argv[i + 2:i + 6]))
    sys.exit(main())


# end of file
