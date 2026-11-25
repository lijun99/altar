#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the time of the likelihoods of kinematicgaussianslip, cpu against cuda, for 1024 samples of one
# front with a rise time per patch: the 24 patches of the tests with 15 stations x 12 bins, and
# 320 patches with 108 stations x 24 bins; gaussian slips and positive dip slips; and the cuda
# likelihoods against the cpu's; with --precision=float32, cuda in float32, the linear algebra in
# float64 and in float32, against the cpu at the samples rounded to float32
#     python benchmark.py <folder> [--precision=float32]
import os
import pathlib
import subprocess
import sys
import time
import numpy

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "trimesh"))

SAMPLES, COMPONENTS, DT = 1024, 3, 2.0
# name: squares along strike and dip, stations, bins
SIZES = {"small": (4, 3, 15, 12), "large": (16, 10, 108, 24)}
CONFIG = """
slipmodel:
    model = altar.models.seismic.kinematicgaussianslip
    model:
        case = ..
        mesh = mesh.h5
        Nmesh = 6
        green = static.gf.h5
        components = {components}
        bins = {bins}
        dt = {dt}
        slip_sigma = 0.7
        {positive}
        {solve}
        dataobs = altar.data.datakron
        dataobs:
            observations = {obs}
            data_file = data.h5
            noise_file = noise.h5
        psets_list = [risetime, rupturevelocity, hypocenter]
        psets:
            risetime = contiguous
            rupturevelocity = contiguous
            hypocenter = contiguous
            risetime.count = {patches}
            rupturevelocity.count = 1
            hypocenter.count = 2
    job:
        tasks = 1
        gpus = {gpus}
        chains = {samples}
        precision = float64
        gpuprecision = {precision}
"""


def setup(folder, size):
    """
    The fault, noise, green's functions, data and samples of {size} in {folder}/{size}
    """
    import h5py
    import kinematic_mesh as km
    import datakron
    NS, ND, stations, bins = SIZES[size]
    km.NS, km.ND = NS, ND
    rng = numpy.random.default_rng(1)
    case = folder / size
    case.mkdir(parents=True, exist_ok=True)
    xyz, uv, faces, sources, owners = km.fault("planar")
    Np = len(faces)
    with h5py.File(case / "mesh.h5", "w") as h5:
        h5["vertices"], h5["uv"], h5["faces"] = xyz, uv, faces
        h5["sources/uv"], h5["sources/patch"] = sources, owners
    datakron.STATIONS, datakron.BINS = stations, bins
    S, R = datakron.noise(rng)
    with h5py.File(case / "noise.h5", "w") as h5:
        h5["S"], h5["R"] = S, R
    with h5py.File(case / "static.gf.h5", "w") as h5:
        h5["static.gf"] = rng.standard_normal((2 * Np, COMPONENTS * stations))
    with h5py.File(case / "data.h5", "w") as h5:
        h5["data"] = rng.standard_normal(COMPONENTS * stations * bins)
    width, depth = 10.0 * NS, 10.0 * ND
    θ = numpy.column_stack([rng.uniform(1, 5, (SAMPLES, Np)), rng.uniform(4, 8, SAMPLES),
                            rng.uniform(0.2, 0.8, SAMPLES) * width, rng.uniform(0.2, 0.8, SAMPLES) * depth])
    numpy.save(case / "theta.npy", θ)
    return Np, stations, bins


def tag(positive, gpu, precision="float64", solve="float64"):
    """
    The name of a run: on the cpu in float64, or at the samples rounded to float32; on cuda, in
    {precision}, the linear algebra in {solve}
    """
    name = f"{'positive' if positive else 'gaussian'}-{'gpu' if gpu else 'cpu'}"
    if precision == "float32":
        name += f"-{precision}-{solve}" if gpu else "-rounded"
    return name


def worker(case, positive, gpu, precision="float64", solve="float64"):
    """
    The likelihoods of the samples, and the time they took, on cuda the best of three; in
    float32 on the cpu, at the samples rounded to float32
    """
    import h5py
    θ = numpy.load(case / "theta.npy")
    if precision == "float32":
        θ = θ.astype(numpy.float32).astype(float)
    Np = θ.shape[1] - 3
    with h5py.File(case / "noise.h5", "r") as h5:
        stations, bins = h5["S"].shape[1], h5["R"].shape[1]
    name = tag(positive, gpu, precision, solve)
    folder = case / name
    folder.mkdir(exist_ok=True)
    (folder / "slipmodel.pfg").write_text(CONFIG.format(
        components=COMPONENTS, bins=bins, dt=DT, positive="positive = dip" if positive else "",
        obs=COMPONENTS * stations * bins, patches=Np, gpus=int(gpu), samples=SAMPLES,
        precision=precision if gpu else "float64", solve=f"solve_precision = {solve}" if gpu else ""))
    os.chdir(folder)
    sys.argv = [sys.argv[0]]
    import altar
    import altar.models.seismic
    model = altar.shells.altar(name="slipmodel").initialize().model
    if gpu:
        import altar.cuda
        theta = altar.cuda.matrix(source=θ, dtype=precision)
        likelihood = altar.cuda.vector(shape=SAMPLES, dtype=precision)
    else:
        theta, likelihood = θ, numpy.zeros(SAMPLES)
    times = []
    for trial in range(3 if gpu else 1):
        start = time.perf_counter()
        model.eval_data_likelihood(theta=theta, likelihood=likelihood)
        values = numpy.array(likelihood, dtype=float)
        times.append(time.perf_counter() - start)
    numpy.savez(case / f"{name}.npz", likelihood=values, time=min(times))
    return 0


def run(folder, size, positive, gpu, precision="float64", solve="float64"):
    """
    A worker in a process of its own; its likelihoods and time
    """
    done = subprocess.run([sys.executable, __file__, "--worker", str(folder / size), str(int(positive)),
                           str(int(gpu)), precision, solve], capture_output=True, text=True)
    if done.returncode:
        print(done.stdout[-2000:], done.stderr[-2000:])
        return None
    return numpy.load(folder / size / f"{tag(positive, gpu, precision, solve)}.npz")


def single(folder):
    """
    cuda in float32, the linear algebra in float64 and in float32, against the cpu at the
    samples rounded to float32
    """
    for size in SIZES:
        Np, stations, bins = setup(folder, size)
        print(f"{size}: {Np} patches, {COMPONENTS} x {stations} x {bins} data, {SAMPLES} samples, float32")
        for positive in (False, True):
            cpu = run(folder, size, positive, False, "float32")
            if cpu is None:
                return 1
            for solve in ("float64", "float32"):
                gpu = run(folder, size, positive, True, "float32", solve)
                if gpu is None:
                    return 1
                error = numpy.abs(gpu["likelihood"] - cpu["likelihood"])
                print(f"  {'positive dip' if positive else 'gaussian':13s} solve {solve}: cuda {float(gpu['time']):8.3f} s; "
                      f"against the cpu, nats: max {error.max():.1e}, median {numpy.median(error):.1e}, "
                      f"relative {error.max() / numpy.abs(cpu['likelihood']).max():.1e}; "
                      f"finite {numpy.isfinite(gpu['likelihood']).sum()} of {SAMPLES}")
    return 0


def main(folder):
    for size in SIZES:
        Np, stations, bins = setup(folder, size)
        print(f"{size}: {Np} patches, {COMPONENTS} x {stations} x {bins} data, {SAMPLES} samples")
        for positive in (False, True):
            out = {}
            for gpu in (False, True):
                done = subprocess.run([sys.executable, __file__, "--worker", str(folder / size), str(int(positive)),
                                       str(int(gpu))], capture_output=True, text=True)
                if done.returncode:
                    print(done.stdout[-2000:], done.stderr[-2000:])
                    return 1
                out[gpu] = numpy.load(folder / size / f"{'positive' if positive else 'gaussian'}-{'gpu' if gpu else 'cpu'}.npz")
            cpu, gpu = out[False], out[True]
            error = numpy.abs(gpu["likelihood"] - cpu["likelihood"]).max() / numpy.abs(cpu["likelihood"]).max()
            print(f"  {'positive dip' if positive else 'gaussian':13s} cpu {float(cpu['time']):8.3f} s, cuda "
                  f"{float(gpu['time']):8.3f} s, x{float(cpu['time']) / float(gpu['time']):6.1f}; "
                  f"cuda against cpu {error:.1e}")
    return 0


if __name__ == "__main__":
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        sys.exit(worker(pathlib.Path(sys.argv[i + 1]), bool(int(sys.argv[i + 2])), bool(int(sys.argv[i + 3])),
                        *sys.argv[i + 4:i + 6]))
    folder = pathlib.Path(sys.argv[1]).resolve()
    sys.exit(single(folder) if "--precision=float32" in sys.argv else main(folder))


# end of file
