#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the kinematic model with its gaussian slips integrated out, kinematicgaussianslip, on a small
# planar fault and gnss-like data of 3 components x 6 stations x 8 bins, on the cpu: the slip
# fractions against the compiled kinematic model, step and bin; the marginal likelihood against
# the dense gaussian of the data, with datakron and with cd_std; a second nucleation point after
# the window, or the same as the first, against one; a sampled slip scale against a fixed one;
# the mean slips and their data; and the samples it rejects. Each configuration runs in a process
# of its own
import os
import pathlib
import subprocess
import sys
import tempfile
import numpy

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trimesh"))

COMPONENTS, STATIONS, BINS, DT, NMESH, CHAINS = 3, 6, 8, 4.0, 6, 6
OBS = COMPONENTS * STATIONS * BINS
CONFIG = """
slipmodel:
    model = altar.models.seismic.kinematicgaussianslip
    model:
        case = ..
        mesh = mesh.h5
        Nmesh = {nmesh}
        green = static.gf.h5
        components = {components}
        bins = {bins}
        dt = {dt}
        integration = {integration}
        slip_sigma = {sigma}
        {extra}
        dataobs = {data}
        dataobs:
            observations = {obs}
            data_file = data.h5
            {covariance}
        psets_list = [{psets}]
        psets:
            risetime = contiguous
            rupturevelocity = contiguous
            hypocenter = contiguous
            origin = contiguous
            log_sigma = contiguous
            risetime:
                count = {rises}
                prior = uniform
                prior.support = (0.5, 10)
            rupturevelocity:
                count = {fronts}
                prior = uniform
                prior.support = (1, 6)
            hypocenter:
                count = {hypocenters}
                prior = uniform
                prior.support = (0, 40)
            origin:
                count = {fronts}
                prior = uniform
                prior.support = (-10, 2000)
            log_sigma:
                count = 1
                prior = uniform
                prior.support = (-2, 2)
    job:
        tasks = 1
        gpus = {gpus}
        chains = {chains}
        precision = float64
        gpuprecision = float64
"""
# name: fronts, rise times, data, integration, sampled scale
RUNS = {
    "one": (1, "patches", "datakron", "bin", False),
    "two": (2, "patches", "datakron", "bin", False),
    "std": (1, 1, "datal2", "step", False),
    "scale": (1, "patches", "datakron", "bin", True),
    "fixed": (1, "patches", "datakron", "bin", False),
    "positive": (1, "patches", "datakron", "bin", False),
}


def worker(case, name, gpu=False):
    """
    The likelihoods, fractions, mean slips and verdicts of the run {name}, into a file; with
    {gpu}, the likelihoods and verdicts on cuda
    """
    fronts, rises, data, integration, scaled = RUNS[name]
    Np = int(numpy.load(case / "patches.npy"))
    rises = Np if rises == "patches" else rises
    psets = ["risetime", "rupturevelocity", "hypocenter"] + ([] if name == "fixed" else ["origin"])
    psets += ["log_sigma"] if scaled else []
    front = numpy.load(case / "front.npy")
    extra = "slip_scale = log_sigma" if scaled else ""
    # without origins, the first chain's, fixed
    extra = f"origin = {float(front[0, Np + 3])!r}" if name == "fixed" else extra
    extra = "positive = dip\n        ghk_paths = 4096" if name == "positive" else extra
    folder = case / (name + ("-gpu" if gpu else ""))
    folder.mkdir()
    (folder / "slipmodel.pfg").write_text(CONFIG.format(
        nmesh=NMESH, components=COMPONENTS, bins=BINS, dt=DT, integration=integration, sigma=0.7,
        extra=extra, data=f"altar.data.{data}", obs=OBS,
        covariance="noise_file = noise.h5" if data == "datakron" else "cd_std = 0.8",
        psets=", ".join(psets), rises=rises, fronts=fronts, hypocenters=2 * fronts, chains=2 * CHAINS,
        gpus=int(gpu)))
    os.chdir(folder)
    sys.argv = [sys.argv[0]]
    import altar
    import altar.models.seismic
    app = altar.shells.altar(name="slipmodel").initialize()
    model = app.model
    Tr, vr, hypo, origin = front[:, :Np], front[:, Np], front[:, Np + 1:Np + 3], front[:, Np + 3]
    rise = Tr if rises == Np else Tr[:, :1]
    if name == "fixed":
        θ = numpy.column_stack([rise, vr, hypo])
    elif fronts == 1:
        θ = numpy.column_stack([rise, vr, hypo, origin])
    else:
        # a second nucleation point after the window, then the same as the first
        late = numpy.column_stack([rise, vr, vr, hypo, hypo, origin, numpy.full(CHAINS, 1500.0)])
        same = numpy.column_stack([rise, vr, vr, hypo, hypo, origin, origin])
        θ = numpy.concatenate([late, same])
    if scaled:
        θ = numpy.column_stack([θ, numpy.full(θ.shape[0], numpy.log10(0.7))])
    if gpu:
        import altar.cuda
        array = lambda x: altar.cuda.matrix(source=x, dtype="float64") if numpy.ndim(x) == 2 \
            else altar.cuda.vector(source=x, dtype="float64")
    else:
        array = lambda x: numpy.array(x, dtype=float)

    def verify(theta):
        mask = array(numpy.zeros(theta.shape[0]))
        model.verify_theta(theta=array(theta), mask=mask)
        return numpy.array(mask)

    likelihood = array(numpy.zeros(θ.shape[0]))
    model.eval_data_likelihood(theta=array(θ), likelihood=likelihood)
    likelihood = numpy.array(likelihood)
    out = model.forward_problem(application=app, theta=θ)
    # the samples to reject: origins out of order, a hypocenter off the mesh
    rejected = {"none": verify(θ)}
    if fronts == 2:
        bad = θ.copy()
        bad[:, model._origin] = bad[:, model._origin][:, ::-1] + [[1.0, 0.0]]
        rejected["order"] = verify(bad)
    off = θ.copy()
    off[:, model._hypocenter[0]] = 39.9
    off[:, model._hypocenter[1]] = 39.9
    rejected["off"] = verify(off)
    numpy.savez(case / f"{name}{'-gpu' if gpu else ''}.npz", likelihood=likelihood, F=out["F"], slip=out["slip"], data=out["data"],
                onsets=model.onsets(θ), **{f"rejected_{k}": v for k, v in rejected.items()})
    return 0


def main():
    gpu = "--gpu" in sys.argv
    import h5py
    from kinematic_mesh import fault
    from altar.data.Kronecker import Kronecker
    from altar.models.seismic.history import SlipHistory
    rng = numpy.random.default_rng(3)
    case = pathlib.Path(tempfile.mkdtemp(prefix="altar-gaussianslip-"))
    xyz, uv, faces, sources, owners = fault("planar")
    Np = len(faces)
    numpy.save(case / "patches.npy", Np)
    with h5py.File(case / "mesh.h5", "w") as h5:
        h5["vertices"], h5["uv"], h5["faces"] = xyz, uv, faces
        h5["sources/uv"], h5["sources/patch"] = sources, owners
    # the noise, as in datakron.py, and the green's functions
    from datakron import noise as make_noise
    import datakron
    datakron.STATIONS, datakron.BINS = STATIONS, BINS
    S, R = make_noise(rng)
    K = Kronecker(S, R)
    with h5py.File(case / "noise.h5", "w") as h5:
        h5["S"], h5["R"] = S, R
    G = rng.standard_normal((2 * Np, COMPONENTS * STATIONS))
    with h5py.File(case / "static.gf.h5", "w") as h5:
        h5["static.gf"] = G
    # the fronts: rise times, velocity, hypocenter, origin
    front = numpy.column_stack([rng.uniform(1, 9, (CHAINS, Np)), rng.uniform(2, 4, CHAINS),
                                rng.uniform(8, 32, CHAINS), rng.uniform(5, 25, CHAINS), rng.uniform(-3, 4, CHAINS)])
    numpy.save(case / "front.npy", front)
    d = rng.standard_normal(OBS)
    with h5py.File(case / "data.h5", "w") as h5:
        h5["data"] = d

    results = {}
    failures = 0
    for name in RUNS:
        done = subprocess.run([sys.executable, __file__, "--worker", str(case), name], capture_output=True, text=True)
        if done.returncode:
            print(done.stdout[-3000:], done.stderr[-3000:])
            failures += 1
            continue
        results[name] = numpy.load(case / f"{name}.npz")
    gpus = {}
    for name in RUNS if gpu else []:
        done = subprocess.run([sys.executable, __file__, "--worker", str(case), name, "--gpu"], capture_output=True,
                              text=True)
        if done.returncode:
            print(done.stdout[-3000:], done.stderr[-3000:])
            failures += 1
            continue
        gpus[name] = numpy.load(case / f"{name}-gpu.npz")
    if failures:
        print("gaussian slip:", f"{failures} FAILED")
        return 1

    def check(label, error, tolerance):
        nonlocal failures
        ok = error <= tolerance
        failures += not ok
        print(f"  {label}: {error:.2e} (tolerance {tolerance:.0e}) {'ok' if ok else 'FAILED'}")

    rel = lambda a, b: numpy.abs(a - b).max() / numpy.abs(b).max()
    one, two, std, scale, fixed, positive = (results[k] for k in RUNS)

    # the onsets against the compiled arrival times plus the origin, the fractions of the model
    # against them, and the fractions against the compiled model's Mb, at origin 0
    from altar.models.seismic.marginal import fractions
    for run, mode in ((one, "bin"), (std, "step")):
        history = SlipHistory(xyz, faces, uv, sources, owners, Nt=BINS, dt=DT, Nmesh=NMESH, integration=mode)
        Tr = front[:, :Np] if mode == "bin" else numpy.repeat(front[:, :1], Np, axis=1)
        velocity = numpy.repeat(front[:, Np:Np + 1], Np, axis=1)
        arrivals, _ = history.arrivals(velocity, front[:, Np + 1:Np + 3])
        onsets = (arrivals + front[:, Np + 3:Np + 4]).reshape(CHAINS, Np, -1)
        check(f"{mode}: onsets against the arrival times plus the origin", numpy.abs(run["onsets"] - onsets).max(), 1e-10)
        check(f"{mode}: fractions against those of the onsets", rel(run["F"], fractions(onsets, Tr, DT, BINS, integration=mode)), 1e-14)
        slip = numpy.zeros((CHAINS, Np, 2))
        slip[..., 0] = 1
        Mb = history(slip, Tr, velocity, front[:, Np + 1:Np + 3])
        compiled = numpy.cumsum(Mb[:, :, 0, :], axis=1).transpose(0, 2, 1)
        mine = fractions(arrivals.reshape(CHAINS, Np, -1), Tr, DT, BINS, integration=mode)
        check(f"{mode}: fractions against the compiled model", rel(mine, compiled), 1e-12)

    # the marginal likelihood against the dense gaussian of the data
    def dense(run, Cd, sigma, fronts=1):
        out = []
        for k, F in enumerate(run["F"]):
            F2 = numpy.concatenate([F, F])
            A = numpy.einsum("jcs,jk->cskj", G.reshape(2 * Np, COMPONENTS, STATIONS), F2).reshape(OBS, 2 * Np)
            C = sigma ** 2 * A @ A.T + Cd
            out.append(-0.5 * (d @ numpy.linalg.solve(C, d) + numpy.linalg.slogdet(C)[1] + OBS * numpy.log(2 * numpy.pi)))
        return numpy.array(out)

    check("datakron: likelihood against the dense gaussian", rel(one["likelihood"], dense(one, K.dense(), 0.7)), 1e-11)
    check("cd_std: likelihood against the dense gaussian", rel(std["likelihood"], dense(std, 0.64 * numpy.eye(OBS), 0.7)), 1e-11)
    check("a second nucleation point after the window, against one", rel(two["likelihood"][:CHAINS], one["likelihood"]), 1e-13)
    check("a second nucleation point the same as the first, against one", rel(two["likelihood"][CHAINS:], one["likelihood"]), 1e-13)
    check("a sampled slip scale against a fixed one", rel(scale["likelihood"], one["likelihood"]), 1e-13)
    check("a fixed origin against the same sampled", abs(fixed["likelihood"][0] - one["likelihood"][0]) / abs(one["likelihood"][0]), 1e-13)

    # positive dip slips: the likelihood gains log Pr(dip slips >= 0 | d) - log Pr(dip slips >= 0),
    # the first against genz's integration, the second 2^-patches; and the mean dip slips positive
    import math
    from scipy.stats import multivariate_normal
    from altar.models.seismic.marginal import GaussianSlip
    marginal = GaussianSlip(G, d, K)
    worst = 0.0
    for k in range(2):
        mean, L = marginal.posterior(one["F"][k:k + 1], 0.7)
        cov = numpy.linalg.inv(L[0] @ L[0].T)[Np:, Np:]
        box = math.log(multivariate_normal(mean=-mean[0, Np:], cov=cov, abseps=1e-9, releps=1e-6,
                                           maxpts=10 ** 7).cdf(numpy.zeros(Np)))
        worst = max(worst, abs(positive["likelihood"][k] - one["likelihood"][k] - box - Np * math.log(2)))
    check("positive dip slips: the likelihood's box probabilities against genz, nats", worst, 2e-2)
    check("positive dip slips: the mean dip slips non-negative", float((positive["slip"][:, Np:] < 0).sum()), 0)

    # the mean slips and their data
    F2 = numpy.concatenate([one["F"], one["F"]], axis=1)
    Ci = numpy.linalg.inv(K.dense())
    worst = 0.0
    for k in range(CHAINS):
        A = numpy.einsum("jcs,jk->cskj", G.reshape(2 * Np, COMPONENTS, STATIONS), F2[k]).reshape(OBS, 2 * Np)
        mean = numpy.linalg.solve(numpy.eye(2 * Np) / 0.49 + A.T @ Ci @ A, A.T @ Ci @ d)
        worst = max(worst, rel(one["slip"][k], mean), rel(one["data"][k], A @ mean))
    check("mean slips and their data against the dense solution", worst, 1e-10)

    # cuda against the cpu, the same sobol points for the bounded slips
    for name, r in gpus.items():
        cpu = results[name]
        check(f"cuda, {name}: likelihood against the cpu", rel(r["likelihood"], cpu["likelihood"]), 1e-10)
        same = all(numpy.array_equal(r[k] != 0, cpu[k] != 0) for k in cpu.files if k.startswith("rejected"))
        check(f"cuda, {name}: the samples rejected, as on the cpu", 0 if same else 1, 0)

    # the samples rejected
    check("valid samples kept", sum(r["rejected_none"].sum() for r in results.values()), 0)
    check("origins out of order rejected", float((two["rejected_order"] == 0).sum()), 0)
    check("hypocenters off the mesh rejected", float(sum((r["rejected_off"] == 0).sum() for r in results.values())), 0)
    print("gaussian slip:", "passed" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        sys.exit(worker(pathlib.Path(sys.argv[i + 1]), sys.argv[i + 2], "--gpu" in sys.argv))
    sys.exit(main())


# end of file
