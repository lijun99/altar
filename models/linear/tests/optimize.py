#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Check the {optimize} action in an mpi run of two processes: on the linear example, patch-9, each
of its methods must find the mean μ of the exact posterior N(μ, C), C = (F + I/σ²)⁻¹, with the
batches of points shared between the processes, and Levenberg-Marquardt the standard deviations
of C at the optimum. With --gpu, both processes share the first gpu.

    python optimize.py
    python optimize.py --gpu
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
# the tolerances of each method on μ, in posterior standard deviations
TOLERANCES = {"cma": 1e-3, "lm": 1e-6, "lbfgs": 1e-2}

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
    optimize:
        posterior = yes
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

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-optimize-"))
    try:
        shutil.copytree(EXAMPLES / CASE, scratch / CASE)
        (scratch / "linear.pfg").write_text(CONFIG)
        (scratch / "plexus.py").write_text(PLEXUS)
        mean, cov = exact()
        sd = numpy.sqrt(numpy.diag(cov))
        failures = 0
        for method, tolerance in TOLERANCES.items():
            command = [sys.executable, "plexus.py", "optimize", "--config=linear.pfg",
                       f"--linear.optimize.method={method}", f"--linear.optimize.output={method}.h5",
                       "--shell=mpi.shells.mpirun", "--job.tasks=2", *devices]
            status = subprocess.run(command, cwd=scratch, capture_output=True, text=True,
                                    errors="replace", timeout=900)
            if status.returncode != 0 or "2 process(es)" not in status.stdout + status.stderr:
                (scratch / f"{method}.log").write_text(status.stdout + status.stderr)
                options.keep = True
                print(f"{method} failed ({status.returncode}); see {scratch}/{method}.log")
                failures += 1
                continue
            with h5py.File(scratch / f"{method}.h5") as h5:
                x = numpy.asarray(h5["best/theta"])
                deviation = numpy.asarray(h5["best/sd"]) if "best/sd" in h5 else None
            offset = (numpy.abs(x - mean) / sd).max()
            checks = [(f"{method}, 2 processes, max |x - μ|/sd", offset, offset <= tolerance)]
            if deviation is not None:
                error = numpy.abs(deviation / sd - 1).max()
                checks.append((f"{method}, 2 processes, sd, max rel error", error, error <= tolerance))
            for name, value, good in checks:
                print(f"{name:40s} {'ok' if good else 'FAIL':4s} {value:12.4g}")
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
