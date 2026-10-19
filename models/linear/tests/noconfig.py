#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Check that an app that reads no configuration file meant for it says so, shows its help screen
and fails, instead of running on its defaults; and that a file named by --config, or one named
after the app, is enough.

    python noconfig.py
"""

import os
import pathlib
import shutil
import subprocess
import tempfile


# the examples, relative to me
EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def run(files, *args):
    """
    Run {altar-linear} with {args} in a scratch directory that holds {files}, and return its exit
    status and everything it said; with an empty user configuration directory, so that the files
    of whoever runs the test don't count
    """
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="altar-noconfig-"))
    try:
        for name, text in files.items():
            (scratch / name).write_text(text)
        # pyre reads {$XDG_CONFIG_HOME/pyre}, if it is there, instead of ~/.config/pyre or ~/.pyre
        (scratch / "config" / "pyre").mkdir(parents=True)
        env = {**os.environ, "XDG_CONFIG_HOME": str(scratch / "config")}
        status = subprocess.run(["altar-linear", *args], cwd=scratch, capture_output=True,
                                text=True, timeout=600, env=env)
        return status.returncode, status.stdout + status.stderr
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    failures = 0

    # no configuration file at all
    code, said = run({})
    missing = [text for text in ("no configuration file was read", "usage:", "--config=<uri>")
               if text not in said]
    ok = code == 1 and not missing
    if not ok:
        print(f"  exit status {code}, missing {missing}")
        print("\n".join(f"    {line}" for line in said.splitlines()[-12:]))
    print(f"no configuration file: {'ok' if ok else 'FAIL'}")
    failures += not ok

    # the example, found either way; the runs need not succeed, only get past the check
    example = (EXAMPLES / "linear.pfg").read_text()
    for label, files, args in [
            ("a file named by --config", {"other.pfg": example}, ["--config=other.pfg"]),
            ("a file named after the app", {"linear.pfg": example}, []),
    ]:
        code, said = run(files, *args, "--job.gpus=0", "--job.tasks=1")
        ok = "no configuration file was read" not in said and "usage:" not in said
        if not ok:
            print("\n".join(f"    {line}" for line in said.splitlines()[-12:]))
        print(f"{label}: {'ok' if ok else 'FAIL'}")
        failures += not ok

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


# end of file
