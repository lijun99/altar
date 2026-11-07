#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Check the {forward} action in an mpi run of two processes: on the linear example, patch-9, with
an odd number of samples of the exact posterior and a reference posterior, the samples shared
unevenly between the processes, its output must be that of a run in one process: the predicted
data, their statistics, and the log likelihood of each sample, in the same order. With --gpu,
both processes share the first gpu.

    python forward.py
    python forward.py --gpu
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy


# the example, relative to me
EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
CASE = "patch-9"
SIGMA = 0.5
# an odd count, so the two processes run different numbers of samples
SAMPLES = 101
# the tolerance on every dataset, relative to its largest value
TOLERANCE = 1e-10
# the datasets to compare
DATASETS = ["theta/mean", "data/mean_model", "data/mean", "data/std", "data/residual",
            "fit/samples/loglikelihood", "fit/mean_model/chi2", "reference/fit/samples/loglikelihood"]

CONFIG = f"""
linear:
    model = altar.models.linear
    model:
        case = {CASE}
        parameters = 18
        dataobs:
            observations = 108
            cd_file = cd.txt
        psets_list = [all]
        psets:
            all = contiguous
            all:
                count = {{linear.model.parameters}}
                prep = gaussian
                prep.sigma = {SIGMA}
                prior = gaussian
                prior.sigma = {SIGMA}
    job.chains = 64
    forward:
        theta = exact.txt
        reference = other.txt
"""

# the plexus of the linear app, as a script, so that mpirun can launch it again in each process
PLEXUS = "import altar, altar.models.linear\nraise SystemExit(altar.shells.altar(name='linear').run())\n"


def exact(folder=EXAMPLES / CASE):
    """
    The mean and the covariance of the posterior for the data in {folder}
    """
    G = numpy.loadtxt(folder / "green.txt")
    d = numpy.loadtxt(folder / "data.txt")
    W = numpy.linalg.inv(numpy.loadtxt(folder / "cd.txt"))
    cov = numpy.linalg.inv(G.T @ W @ G + numpy.eye(G.shape[1]) / SIGMA**2)
    return cov @ (G.T @ W @ d), cov


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gpu", action="store_true", help="run both processes on the first gpu")
    parser.add_argument("--keep", action="store_true", help="keep the scratch directory")
    options = parser.parse_args()
    devices = ["--job.gpus=1", "--job.gpuids=[0, 0]"] if options.gpu else ["--job.gpus=0"]

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-forward-"))
    try:
        shutil.copytree(EXAMPLES / CASE, scratch / CASE)
        (scratch / "linear.pfg").write_text(CONFIG)
        (scratch / "plexus.py").write_text(PLEXUS)
        mean, cov = exact()
        rng = numpy.random.default_rng(1)
        numpy.savetxt(scratch / "exact.txt", rng.multivariate_normal(mean, cov, size=SAMPLES))
        numpy.savetxt(scratch / "other.txt", rng.multivariate_normal(mean, 2 * cov, size=SAMPLES))
        runs = {"one": [devices[0]] + (["--job.gpuids=[0]"] if options.gpu else []),
                "two": ["--shell=mpi.shells.mpirun", "--job.tasks=2", *devices]}
        for name, extra in runs.items():
            command = [sys.executable, "plexus.py", "forward", "--config=linear.pfg",
                       f"--linear.forward.output={name}.h5", *extra]
            status = subprocess.run(command, cwd=scratch, capture_output=True, text=True,
                                    errors="replace", timeout=900)
            processes = "2 process(es)" if name == "two" else "1 process(es)"
            if status.returncode != 0 or processes not in status.stdout + status.stderr:
                (scratch / f"{name}.log").write_text(status.stdout + status.stderr)
                options.keep = True
                print(f"{name} failed ({status.returncode}); see {scratch}/{name}.log")
                return 1
        failures = 0
        with h5py.File(scratch / "one.h5") as one, h5py.File(scratch / "two.h5") as two:
            for key in DATASETS:
                a, b = numpy.asarray(one[key], dtype=float), numpy.asarray(two[key], dtype=float)
                error = numpy.abs(a - b).max() / max(numpy.abs(a).max(), 1e-300) \
                    if a.shape == b.shape else numpy.inf
                good = error <= TOLERANCE
                print(f"2 processes vs 1, {key:36s} {'ok' if good else 'FAIL':4s} {error:12.4g}")
                failures += not good
        return 1 if failures else 0
    finally:
        if options.keep:
            print(f"  kept {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())


# end of file
