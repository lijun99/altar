#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Check that pyre reads altar configurations in yaml: a yaml translation of the linear example
configures the app exactly as {linear.pfg} does, and {linear_gpu.yaml} sets what it says.

    python config.py
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile


# the examples, relative to me
EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"

# linear.pfg in yaml, with the component choices in blocks of their own since yaml keys can't repeat
YAML = """
linear:
  model: altar.models.linear
  job.tasks: 1
  job.gpus: 0
  job.chains: 2**8
  job.steps: 2**8
linear.model:
  case: patch-9
  parameters: 18
  dataobs:
    observations: 108
    cd_file: cd.txt
  psets_list: [all]
  psets:
    all: contiguous
linear.model.psets.all:
  count: "{linear.model.parameters}"
  prep: gaussian
  prep.sigma: 0.5
  prior: gaussian
  prior.sigma: 0.5
"""

# print every setting of the linear app, walking into its components, without running it
DUMP = """
import altar, pyre

class Linear(altar.shells.application, family="altar.applications.linear"):
    model = altar.models.model(default="linear")

def walk(component, path, seen):
    if id(component) in seen:
        return
    seen.add(id(component))
    print(f"{path} = {type(component).__name__}")
    for trait in component.pyre_configurables():
        value = getattr(component, trait.name)
        key = f"{path}.{trait.name}"
        if isinstance(value, pyre.component):
            walk(value, key, seen)
        elif hasattr(value, "items") and all(isinstance(v, pyre.component) for _, v in value.items()):
            for name, v in value.items():
                walk(v, f"{key}.{name}", seen)
        else:
            print(f"{key} = {value!r}")

walk(Linear(name="linear"), "linear", set())
"""


def settings(files, *args):
    """
    The settings of the linear app configured by {files}, copied into a scratch directory
    """
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-config-"))
    try:
        for name, text in files.items():
            (scratch / name).write_text(text)
        status = subprocess.run([sys.executable, "-c", DUMP, *args], cwd=scratch,
                                capture_output=True, text=True, check=True)
        # the settings, without any journal output, e.g. the warning that there is no gpu
        return dict(line.split(" = ", 1) for line in status.stdout.splitlines()
                    if line.startswith("linear") and " = " in line)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    failures = 0

    pfg = settings({"linear.pfg": (EXAMPLES / "linear.pfg").read_text()})
    yaml = settings({"linear.yaml": YAML})
    differ = sorted(key for key in pfg.keys() | yaml.keys() if pfg.get(key) != yaml.get(key))
    for key in differ:
        print(f"  {key}: pfg {pfg.get(key)}, yaml {yaml.get(key)}")
    print(f"linear.pfg and its yaml translation: {'FAIL' if differ else 'ok'}, {len(pfg)} settings")
    failures += bool(differ)

    gpu = settings({"linear_gpu.yaml": (EXAMPLES / "linear_gpu.yaml").read_text()},
                   "--config=linear_gpu.yaml")
    expected = {
        "linear.job.gpus": "1",
        "linear.job.gpuprecision": "'float64'",
        "linear.job.gpuids": "[0]",
        "linear.job.chains": "1024",
        "linear.controller": "Catmip",
        "linear.controller.archiver": "H5Recorder",
        "linear.model.psets.all.count": "18",
        "linear.model.psets.all.prior.sigma": "0.5",
    }
    wrong = {key: gpu.get(key) for key, value in expected.items() if gpu.get(key) != value}
    for key, value in wrong.items():
        print(f"  {key}: {value}, expected {expected[key]}")
    print(f"linear_gpu.yaml: {'FAIL' if wrong else 'ok'}")
    failures += bool(wrong)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


# end of file
