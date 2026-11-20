#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# kinematicgaussianslip against kinematicmesh, which samples the slips, on one synthetic: the same
# fault, data, noise, priors and source time function (bin), so the same posterior; the front,
# the slips and the evidence of each run, and the cost; with "static", static data of the slips too,
# with ramps: kinematicgaussianslip with them against an ensemble of the static model and
# kinematicmesh, kmjoint, annealed together, the ramps uniform there
#     python compare.py setup <folder> [gaussian|positive|static]
#     cd <folder>/<run> && slipmodel --config=slipmodel.pfg     for run in gs, km, kmhmc
#     python compare.py evaluate <folder> <run>
#     python compare.py report <folder>
import os
import pathlib
import sys
import numpy

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "trimesh"))

COMPONENTS, STATIONS, BINS, DT, NMESH, SIGMA = 3, 15, 12, 2.0, 6, 0.7
RISE, VELOCITY, HYPOCENTER = 3.0, 5.0, (10.0, 15.0)
# the static data: observations, ramps, the support of the uniform prior of each ramp in kmjoint
STATIC, RAMPS, RAMP = 60, 3, 20.0
STATIC_GS = """
        static_green = static_insar.gf.h5
        static_ramps = {ramps}
        static_data = altar.data.datal2
        static_data:
            observations = {static}
            data_file = static.h5
            cd_file = static.cd.h5"""
DATA = """
        dataobs = altar.data.datakron
        dataobs:
            observations = {obs}
            data_file = data.h5
            noise_file = noise.h5"""
FRONT = """
            risetime:
                count = {patches}
                prior = uniform
                prior.support = (0.5, 10)
                prior.reparameterize = {reparameterize}
            rupturevelocity:
                count = 1
                prior = uniform
                prior.support = (1, 15)
                prior.reparameterize = {reparameterize}
            hypocenter:
                count = 2
                prior = uniform
                prior.support = (0, 30)
                prior.reparameterize = {reparameterize}"""
GS = """
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
        slip_sigma = {sigma}
        {positive}{static_gs}""" + DATA + """
        psets_list = [risetime, rupturevelocity, hypocenter]
        psets:
            risetime = contiguous
            rupturevelocity = contiguous
            hypocenter = contiguous""" + FRONT + """
    controller = altar.bayesian.catmip
    controller.archiver = altar.bayesian.h5recorder
    controller.archiver.output_dir = results
    job:
        tasks = 1
        gpus = 0
        chains = {chains}
        steps = {steps}
        precision = float64
"""
KM = """
slipmodel:
    model = altar.models.seismic.kinematicmesh
    model:
        case = ..
        mesh = mesh.h5
        Nmesh = {nmesh}
        velocity_groups = groups
        green = kinematicG.gf.h5
        Nt = {bins}
        dt = {dt}
        integration = bin""" + DATA + """
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
                prior.sigma = {sigma}
            dipslip:
                count = {patches}{dipslip}""" + FRONT + """
    controller = altar.bayesian.{controller}
    controller.archiver = altar.bayesian.h5recorder
    controller.archiver.output_dir = results
    job:
        tasks = 1
        gpus = 1
        chains = {chains}
        steps = {steps}
        precision = float64
        gpuprecision = float64
"""
KMJOINT = """
slipmodel:
    model = altar.models.ensemble
    model:
        case = ..
        psets_list = [strikeslip, dipslip, ramp, risetime, rupturevelocity, hypocenter]
        psets:
            strikeslip = contiguous
            dipslip = contiguous
            ramp = contiguous
            risetime = contiguous
            rupturevelocity = contiguous
            hypocenter = contiguous
            strikeslip:
                count = {patches}
                prior = gaussian
                prior.sigma = {sigma}
            dipslip:
                count = {patches}{dipslip}
            ramp:
                count = {ramps}
                prior = uniform
                prior.support = (-{ramp}, {ramp})""" + FRONT + """
        models:
            static = altar.models.seismic.static
            kinematic = altar.models.seismic.kinematicmesh
            static:
                case = ..
                psets_list = [strikeslip, dipslip, ramp]
                green = static_insar.gf.h5
                dataobs:
                    observations = {static}
                    data_file = static.h5
                    cd_file = static.cd.h5
            kinematic:
                case = ..
                mesh = mesh.h5
                Nmesh = {nmesh}
                velocity_groups = groups
                green = kinematicG.gf.h5
                Nt = {bins}
                dt = {dt}
                integration = bin
                psets_list = [strikeslip, dipslip, risetime, rupturevelocity, hypocenter]
                dataobs = altar.data.datakron
                dataobs:
                    observations = {obs}
                    data_file = data.h5
                    noise_file = noise.h5
    controller = altar.bayesian.{controller}
    controller.archiver = altar.bayesian.h5recorder
    controller.archiver.output_dir = results
    job:
        tasks = 1
        gpus = 1
        chains = {chains}
        steps = {steps}
        precision = float64
        gpuprecision = float64
"""
# run: template, controller, chains, steps, reparameterized
RUNS = {"gs": (GS, "catmip", 2048, 50, False), "km": (KM, "catmip", 4096, 100, False),
        "kmhmc": (KM, "catmip_hmc", 1024, 20, True)}
# with static data
STATIC_RUNS = {"gs": RUNS["gs"], "kmjoint": (KMJOINT, "catmip", 4096, 100, False)}


# the prior of the dip slips of kinematicmesh: gaussian, or truncated to positive slips
DIPSLIP = {
    "gaussian": """
                prior = gaussian
                prior.sigma = {sigma}""",
    "static": """
                prior = gaussian
                prior.sigma = {sigma}""",
    "positive": """
                prior = tgaussian
                prior.sigma = {sigma}
                prior.support = (0, 100)
                prior.reparameterize = {reparameterize}"""}


def setup(folder, prior="gaussian"):
    """
    The inputs, the truth and the configurations of the runs, in {folder}; with {prior}
    "positive", the dip slips bounded below by 0 in the truth and the priors; with "static",
    static data with ramps too
    """
    import h5py
    from kinematic_mesh import fault
    import datakron
    folder.mkdir(parents=True, exist_ok=True)
    rng = numpy.random.default_rng(2026)
    xyz, uv, faces, sources, owners = fault("planar")
    Np = len(faces)
    obs = COMPONENTS * STATIONS * BINS
    with h5py.File(folder / "mesh.h5", "w") as h5:
        h5["vertices"], h5["uv"], h5["faces"] = xyz, uv, faces
        h5["sources/uv"], h5["sources/patch"] = sources, owners
        h5["groups"] = numpy.zeros(Np, dtype=int)
    datakron.STATIONS, datakron.BINS = STATIONS, BINS
    S, R = datakron.noise(rng)
    with h5py.File(folder / "noise.h5", "w") as h5:
        h5["S"], h5["R"] = S, R
    G = rng.standard_normal((2 * Np, COMPONENTS * STATIONS))
    with h5py.File(folder / "static.gf.h5", "w") as h5:
        h5["static.gf"] = G
    # the big G of kinematicmesh: the slip of interval i moves the data of bins i on
    big = numpy.zeros((BINS, 2 * Np, COMPONENTS * STATIONS, BINS))
    for i in range(BINS):
        big[i, :, :, i:] = G[:, :, None]
    with h5py.File(folder / "kinematicG.gf.h5", "w") as h5:
        h5["kinematicG.gf"] = big.reshape(BINS * 2 * Np, obs)
    with h5py.File(folder / "data.h5", "w") as h5:
        h5["data"] = numpy.zeros(obs)
    # the static data: green's functions of the slips then of the ramps, a dense covariance
    Gs = rng.standard_normal((STATIC, 2 * Np + RAMPS))
    X = rng.standard_normal((STATIC, STATIC))
    Cs = 0.2 * X @ X.T / STATIC + 0.3 * numpy.eye(STATIC)
    for file, value in (("static_insar.gf.h5", Gs), ("static.cd.h5", Cs), ("static.h5", numpy.zeros(STATIC))):
        with h5py.File(folder / file, "w") as h5:
            h5["data"] = value
    runs = STATIC_RUNS if prior == "static" else RUNS
    for run, (template, controller, chains, steps, reparameterize) in runs.items():
        (folder / run).mkdir(exist_ok=True)
        dipslip = DIPSLIP[prior].format(sigma=SIGMA, reparameterize=reparameterize)
        (folder / run / "slipmodel.pfg").write_text(template.format(
            nmesh=NMESH, components=COMPONENTS, bins=BINS, dt=DT, sigma=SIGMA, obs=obs, patches=Np,
            controller=controller, chains=chains, steps=steps, reparameterize=reparameterize,
            positive="positive = dip" if prior == "positive" else "", dipslip=dipslip,
            static_gs=STATIC_GS.format(ramps=RAMPS, static=STATIC) if prior == "static" else "",
            static=STATIC, ramps=RAMPS, ramp=RAMP))
    # the data of the truth, by kinematicgaussianslip
    model = application(folder / "gs").model
    from altar.data.Kronecker import Kronecker
    front = numpy.concatenate([numpy.full(Np, RISE), [VELOCITY], HYPOCENTER])
    slip = rng.normal(0, SIGMA, 2 * Np)
    if prior == "positive":
        slip[Np:] = numpy.abs(slip[Np:])
    signal = model.marginal.predict(model.fractions(front[None]), slip[None])[0]
    d = signal + Kronecker(S, R).draw(rng)
    with h5py.File(folder / "data.h5", "w") as h5:
        h5["data"] = d
    ramps = rng.uniform(-3, 3, RAMPS)
    static = Gs @ numpy.concatenate([slip, ramps])
    with h5py.File(folder / "static.h5", "w") as h5:
        h5["data"] = static + numpy.linalg.cholesky(Cs) @ rng.standard_normal(STATIC)
    numpy.savez(folder / "truth.npz", front=front, slip=slip, signal=signal, ramps=ramps, static=static)
    print(f"setup: {Np} patches, {obs} data, signal rms {numpy.sqrt((signal ** 2).mean()):.2f}")
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


def evaluate(folder, run):
    """
    The front, (samples, rise times + velocity + hypocenter), the slips, (samples, 2 patches),
    and the log evidence of {run}, into {folder}/{run}.npz
    """
    import h5py
    app = application(folder / run)
    model = app.model
    from altar.actions.Forward import load_samples
    θ = load_samples(model=model, path=folder / run / "results" / "step_final.h5")
    with h5py.File(folder / run / "results" / "step_final.h5", "r") as h5:
        evidence = float(numpy.asarray(h5["Annealer/log_evidence"]).ravel()[-1])
    truth = numpy.load(folder / "truth.npz")
    Np = truth["slip"].size // 2
    out = {"evidence": evidence}
    if run.startswith("gs"):
        out["front"] = θ[:, :Np + 3]
        out["slip"] = model.draw_slips(θ, numpy.random.default_rng(1))[:, 0, :]
    elif run == "kmjoint":
        # the evidence of flat ramps, as gs's: without the uniform prior's density, (2 RAMP)^-RAMPS
        out["evidence"] = evidence + RAMPS * numpy.log(2 * RAMP)
        out["front"] = θ[:, 2 * Np + RAMPS:]
        out["slip"] = θ[:, :2 * Np]
        out["ramps"] = θ[:, 2 * Np:2 * Np + RAMPS]
    else:
        out["front"] = θ[:, 2 * Np:]
        out["slip"] = θ[:, :2 * Np]
        # the same data model: kinematicmesh's data of the truth against those it was made with
        θtrue = numpy.concatenate([truth["slip"], truth["front"]])[None]
        data = model.forward_problem(application=app, theta=θtrue)["data"][0]
        out["signal"] = numpy.abs(data - truth["signal"]).max() / numpy.abs(truth["signal"]).max()
    numpy.savez(folder / f"{run}.npz", **out)
    return 0


def density(folder):
    """
    The marginal log likelihood of the fronts, the slips integrated out, at the samples of each
    run and at the truth; the priors of the fronts are uniform, so these rank the posterior density
    """
    model = application(folder / "gs").model
    truth = numpy.load(folder / "truth.npz")
    rows = {"truth": truth["front"][None]}
    for path in sorted(folder.glob("*.npz")):
        if path.stem != "truth":
            rows[path.stem] = numpy.load(path)["front"]
    print(f"{'':8s}{'samples':>8s}{'median':>10s}{'max':>10s}{'10%':>10s}{'90%':>10s}")
    for name, front in rows.items():
        θ = front
        L = numpy.zeros(θ.shape[0])
        model.eval_data_likelihood(theta=θ, likelihood=L)
        print(f"{name:8s}{L.size:8d}{numpy.median(L):10.2f}{L.max():10.2f}"
              f"{numpy.percentile(L, 10):10.2f}{numpy.percentile(L, 90):10.2f}")
    return 0


def report(folder):
    """
    The runs side by side
    """
    truth = numpy.load(folder / "truth.npz")
    Np = truth["slip"].size // 2
    names = [run for run in {**RUNS, **STATIC_RUNS} if (folder / f"{run}.npz").exists()]
    names += sorted(p.stem for p in folder.glob("gs-*.npz"))
    runs = {run: numpy.load(folder / f"{run}.npz") for run in names}
    times = {}
    for run in runs:
        log = folder / run / "time.txt"
        times[run] = log.read_text().strip() if log.exists() else "?"
    if "km" in runs:
        print(f"the data model of kinematicmesh against kinematicgaussianslip's: {float(runs['km']['signal']):.1e}")
    print(f"{'':18s}" + "".join(f"{run:>24s}" for run in runs))
    labels = [("velocity", Np), ("hypocenter u", Np + 1), ("hypocenter v", Np + 2)]
    for label, column in labels:
        cells = "".join(f"{r['front'][:, column].mean():12.3f} ± {r['front'][:, column].std():9.3f}" for r in runs.values())
        print(f"{label:12s}{truth['front'][column]:6.2f}{cells}")
    cells = "".join(f"{r['front'][:, :Np].mean():12.3f} ± {r['front'][:, :Np].mean(axis=1).std():9.3f}" for r in runs.values())
    print(f"{'mean rise':12s}{RISE:6.2f}{cells}")
    print(f"{'log evidence':18s}" + "".join(f"{float(r['evidence']):24.2f}" for r in runs.values()))
    if "kmjoint" in runs:
        print(f"  (kmjoint's with the density of its uniform ramp prior taken out, for flat ramps as gs's)")
        print(f"ramps {numpy.round(truth['ramps'], 3)}: kmjoint {numpy.round(runs['kmjoint']['ramps'].mean(axis=0), 3)} "
              f"± {numpy.round(runs['kmjoint']['ramps'].std(axis=0), 3)}")
    print(f"{'samples':18s}" + "".join(f"{r['front'].shape[0]:24d}" for r in runs.values()))
    print(f"{'time':18s}" + "".join(f"{times[run]:>24s}" for run in runs))
    # the slips and the rise times against those of gs
    if "gs" in runs:
        ref = runs["gs"]
        m0, s0 = ref["slip"].mean(axis=0), ref["slip"].std(axis=0)
        r0m, r0s = ref["front"][:, :Np].mean(axis=0), ref["front"][:, :Np].std(axis=0)
        for run, r in runs.items():
            m, s = r["slip"].mean(axis=0), r["slip"].std(axis=0)
            rm, rs = r["front"][:, :Np].mean(axis=0), r["front"][:, :Np].std(axis=0)
            z = (slip := truth["slip"]) - m
            print(f"{run:6s} slips: corr with the truth {numpy.corrcoef(m, slip)[0, 1]:.3f}, within 2 sd "
                  f"{(numpy.abs(z) <= 2 * s).mean():.0%}; against gs: |mean diff|/sd {numpy.median(numpy.abs(m - m0) / s0):.2f} "
                  f"(median), sd ratio {numpy.median(s / s0):.2f}; rise times against gs: |mean diff|/sd "
                  f"{numpy.median(numpy.abs(rm - r0m) / r0s):.2f}, sd ratio {numpy.median(rs / r0s):.2f}")
    return 0


if __name__ == "__main__":
    command, folder = sys.argv[1], pathlib.Path(sys.argv[2]).resolve()
    if command == "evaluate":
        sys.exit(evaluate(folder, sys.argv[3]))
    if command == "setup":
        sys.exit(setup(folder, *sys.argv[3:4]))
    sys.exit({"setup": setup, "report": report, "density": density}[command](folder))


# end of file
