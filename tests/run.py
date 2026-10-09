#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Run the tests of an installed AlTar from this source tree: the cpu suite, or the gpu suite,
for a machine with gpus. A test passes when its command exits
with 0; an example run must also take its annealing to beta = 1. Tests that need something
optional are skipped without it: jax, cuTile, the seismic 9patch data, or two gpus.

    python tests/run.py                          # the cpu suite
    python tests/run.py --gpu                    # the gpu suite
    python tests/run.py --list                   # the tests of a suite, with what they need
    python tests/run.py linear posterior         # the tests whose names contain these words
    python tests/run.py --gpu --seismic-data ~/data/9patch
"""

import argparse
import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from importlib.util import find_spec


# the source tree, relative to me
ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


@dataclasses.dataclass
class Test:
    """
    A command to run, in a directory, and what it needs
    """
    name: str
    command: list
    cwd: pathlib.Path = ROOT
    needs: tuple = ()
    # the examples directory to copy into a scratch directory and run in, if any
    examples: str | None = None
    # the synthetic data generator to run first, in the copy, if any
    synthetic: str | None = None
    # whether the run must reach beta = 1
    anneals: bool = False
    timeout: int = 1800


def scripts(directory, names, needs=()):
    """
    Tests that run the python scripts {names} of {directory}, each in its own directory
    """
    return [Test(name=f"{directory.relative_to(ROOT)}/{name}".replace("/", ":").replace(".py", ""),
                 command=[sys.executable, name], cwd=directory, needs=needs) for name in names]


def example(app, model, config, gpus, *settings, needs=(), synthetic=None, variant=None):
    """
    A short run of an example, with few chains and steps, in a scratch copy of its directory
    """
    name = f"example:{model}:{config.rsplit('.', 1)[0]}" + (f":{variant}" if variant else "")
    command = [app, f"--config={config}", f"--job.gpus={gpus}", "--job.tasks=1",
               "--job.chains=64", "--job.steps=10", "--controller.archiver.output_dir=results",
               *settings]
    return Test(name=name, command=command, examples=model, synthetic=synthetic, needs=needs,
                anneals=True)


def cpu():
    """
    The cpu suite
    """
    tests = scripts(ROOT / "altar/tests/altar", ["sanity.py", "application.py", "application_instance.py",
                                                  "annealer.py", "annealer_instance.py",
                                                  "weighted_statistics.py", "outliers.py"])
    tests += scripts(MODELS / "cdm/tests", ["sanity.py", "libcdm.py"])
    tests += scripts(MODELS / "linear/tests", ["config.py"])
    tests += [
        example("altar-linear", "linear", "linear.pfg", 0),
        example("altar-linear", "linear", "linear_hmc.pfg", 0),
        example("altar-linear", "linear", "linear_mala.pfg", 0),
        example("altar-linear", "linear", "linear_mcmc.pfg", 0),
        example("altar-regression", "regression", "linear.pfg", 0),
        example("altar-regression", "regression", "linear_hmc.pfg", 0),
        example("altar-mogi", "mogi", "mogi.pfg", 0, synthetic="mogi.py"),
        example("altar-cdm", "cdm", "cdm.pfg", 0, synthetic="cdm.py"),
        example("altar-reverso", "reverso", "reverso.pfg", 0, synthetic="reverso.py"),
        example("gaussian", "gaussian", "gaussian.pfg", 0),
        example("slipmodel", "seismic", "static.pfg", 0, needs=("seismic-data",)),
        example("slipmodel", "seismic", "static.pfg", 0, "--controller=altar.bayesian.cf_catmip",
                needs=("seismic-data",), variant="cf_catmip"),
    ]
    tests.append(Test(name="example:linear:mpi", needs=("mpi",), examples="linear", anneals=True,
                      command=["altar-linear", "--config=linear.pfg", "--job.gpus=0", "--job.tasks=2",
                               "--shell=mpi.shells.mpirun", "--job.chains=64", "--job.steps=10",
                               "--controller.archiver.output_dir=results"]))
    for precision in ["float64", "float32"]:
        tests.append(Test(name=f"posterior:linear:cpu:{precision}", cwd=MODELS / "linear/tests",
                          command=[sys.executable, "posterior.py", f"--precision={precision}"]))
    return tests


def gpu():
    """
    The gpu suite
    """
    cuda = sorted(path.name for path in (ROOT / "altar/tests/cuda").glob("*.py"))
    tests = scripts(ROOT / "altar/tests/cuda", cuda)
    tests += scripts(MODELS / "linear/tests", ["config.py"])
    tests += [
        example("altar-linear", "linear", "linear.pfg", 1),
        example("altar-linear", "linear", "linear_hmc.pfg", 1),
        example("altar-linear", "linear", "linear_mala.pfg", 1),
        example("altar-linear", "linear", "linear_mcmc.pfg", 1),
        example("altar-mogi", "mogi", "mogi.pfg", 1, synthetic="mogi.py", needs=("cutile",)),
        example("altar-cdm", "cdm", "cdm.pfg", 1, synthetic="cdm.py", needs=("cutile",)),
        example("altar-reverso", "reverso", "reverso.pfg", 1, synthetic="reverso.py", needs=("cutile",)),
        example("slipmodel", "seismic", "static.pfg", 1, needs=("seismic-data",)),
        example("slipmodel", "seismic", "static.pfg", 1, "--controller=altar.bayesian.cf_catmip",
                needs=("seismic-data",), variant="cf_catmip"),
        example("slipmodel", "seismic", "kinematic.pfg", 1, needs=("seismic-data",)),
    ]
    tests.append(Test(name="example:linear:mpi-2gpus", needs=("mpi", "2gpus"), examples="linear",
                      anneals=True,
                      command=["altar-linear", "--config=linear.pfg", "--job.gpus=1", "--job.tasks=2",
                               "--shell=mpi.shells.mpirun", "--job.chains=64", "--job.steps=10",
                               "--controller.archiver.output_dir=results"]))
    for precision in ["float64", "float32"]:
        tests.append(Test(name=f"posterior:linear:gpu:{precision}", cwd=MODELS / "linear/tests",
                          command=[sys.executable, "posterior.py", "--gpu", f"--precision={precision}"]))
        tests.append(Test(name=f"kinematic:gradient-vs-jax:{precision}", examples="seismic",
                          needs=("jax", "seismic-data"),
                          command=[sys.executable, str(MODELS / "seismic/tests/kinematic_jax.py"),
                                   "--config=kinematic.pfg", f"--job.gpuprecision={precision}",
                                   "--job.chains=8"]))
    return tests


def missing(test, options):
    """
    The first of the needs of {test} this machine doesn't meet, if any
    """
    for need in test.needs:
        if need == "jax" and find_spec("jax") is None:
            return "jax is not installed"
        if need == "cutile" and (find_spec("cuda") is None or find_spec("cuda.tile") is None):
            return "cuTile is not installed"
        if need == "seismic-data" and not (options.seismic_data and options.seismic_data.is_dir()):
            return "no seismic 9patch data; see --seismic-data"
        if need == "mpi" and shutil.which("mpirun") is None:
            return "mpirun is not on the PATH"
        if need == "2gpus" and gpu_count() < 2:
            return "fewer than two gpus"
    return None


def gpu_count():
    """
    The number of visible gpus, as nvidia-smi sees them
    """
    if shutil.which("nvidia-smi") is None:
        return 0
    listing = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.split()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return len(listing) if visible is None else len([g for g in visible.split(",") if g.strip()])


def run(test, options, env):
    """
    Run {test}; return its verdict, its time, and a line about it
    """
    scratch = None
    cwd = test.cwd
    try:
        if test.examples:
            scratch = pathlib.Path(tempfile.mkdtemp(prefix=f"altar-test-{test.examples}-"))
            shutil.copytree(MODELS / test.examples / "examples", scratch, dirs_exist_ok=True)
            if test.examples == "seismic" and options.seismic_data:
                shutil.copytree(options.seismic_data, scratch / "9patch", dirs_exist_ok=True)
            if test.synthetic:
                subprocess.run([sys.executable, test.synthetic], cwd=scratch / "synthetic", env=env,
                               capture_output=True, check=True, timeout=test.timeout)
            cwd = scratch
        start = time.perf_counter()
        status = subprocess.run(test.command, cwd=cwd, env=env, capture_output=True, text=True,
                                errors="replace", timeout=test.timeout)
        elapsed = time.perf_counter() - start
        log = status.stdout + status.stderr
        good = status.returncode == 0
        detail = f"exit {status.returncode}"
        if good and test.anneals:
            betas = [line.rsplit("beta:", 1)[1].split(",")[0].strip()
                     for line in log.splitlines() if "iteration:" in line and "beta:" in line]
            good = bool(betas) and float(betas[-1]) == 1.0
            detail = f"{len(betas)} beta steps, last beta {betas[-1] if betas else '-'}"
        if not good:
            options.logs.mkdir(parents=True, exist_ok=True)
            (options.logs / f"{test.name.replace(':', '_')}.log").write_text(log)
            lines = [line for line in log.splitlines() if line.strip()]
            detail += f"; {lines[-1][:100] if lines else 'no output'}"
        return good, elapsed, detail
    except subprocess.TimeoutExpired:
        return False, test.timeout, f"timed out after {test.timeout}s"
    except subprocess.CalledProcessError as error:
        return False, 0.0, f"making the synthetic data failed: {error}"
    finally:
        if scratch and not options.keep:
            shutil.rmtree(scratch, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("words", nargs="*", help="run only the tests whose names contain these")
    parser.add_argument("--gpu", action="store_true", help="the gpu suite instead of the cpu one")
    parser.add_argument("--list", action="store_true", help="list the tests and what they need")
    parser.add_argument("--keep", action="store_true", help="keep the scratch directories")
    parser.add_argument("--logs", type=pathlib.Path, default=pathlib.Path("test-logs"),
                        help="where the logs of failed tests go")
    parser.add_argument("--seismic-data", type=pathlib.Path,
                        default=os.environ.get("ALTAR_SEISMIC_DATA"),
                        help="the directory with the seismic 9patch inputs; $ALTAR_SEISMIC_DATA")
    options = parser.parse_args()

    tests = gpu() if options.gpu else cpu()
    if options.words:
        tests = [test for test in tests if any(word in test.name for word in options.words)]
    if options.list:
        for test in tests:
            print(f"{test.name:42s} {', '.join(test.needs) or '-'}")
        return 0

    # the cpu suite runs with no gpu in sight, as on the github runners
    env = dict(os.environ)
    if not options.gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    counts = {"ok": 0, "FAIL": 0, "skip": 0}
    for test in tests:
        reason = missing(test, options)
        if reason:
            counts["skip"] += 1
            print(f"{test.name:42s} skip          {reason}", flush=True)
            continue
        good, elapsed, detail = run(test, options, env)
        verdict = "ok" if good else "FAIL"
        counts[verdict] += 1
        print(f"{test.name:42s} {verdict:4s} {elapsed:6.1f}s  {detail}", flush=True)
    print(f"{counts['ok']} passed, {counts['FAIL']} failed, {counts['skip']} skipped")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())


# end of file
