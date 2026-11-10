#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Compare the final samples of two runs, a reference and another: the shift of each mean in the
reference's standard deviations, the ratio of the standard deviations, the largest difference of
the correlations, and the log evidence of each

    python compare.py results/static_perturbation/step_final.h5 results/static_catmip/step_final.h5
"""

import sys
import h5py
import numpy


def load(path):
    """
    The physical samples of the final step, its parameter sets side by side, and its evidence
    """
    with h5py.File(path) as h5:
        sets = h5["ParameterSets"]
        names = sorted({key.rsplit("_", 1)[0] for key in sets if key.endswith(("_physical", "_sampling"))})
        blocks = [numpy.asarray(sets[f"{name}_physical" if f"{name}_physical" in sets else f"{name}_sampling"],
                                dtype=float) for name in names]
        evidence = h5["Annealer"].get("log_evidence")
        return names, numpy.hstack(blocks), None if evidence is None else float(numpy.asarray(evidence))


def main(reference, other):
    names, a, ea = load(reference)
    _, b, eb = load(other)
    sd = a.std(axis=0)
    shift = (b.mean(axis=0) - a.mean(axis=0)) / sd
    ratio = b.std(axis=0) / sd
    drift = numpy.abs(numpy.corrcoef(a, rowvar=False) - numpy.corrcoef(b, rowvar=False)).max()
    print(f"parameter sets {', '.join(names)}: {a.shape[1]} parameters, {a.shape[0]} and {b.shape[0]} samples")
    print(f"mean shift: max {numpy.abs(shift).max():.3f}, rms {numpy.sqrt((shift ** 2).mean()):.3f} sd")
    print(f"sd ratio: [{ratio.min():.3f}, {ratio.max():.3f}], mean {ratio.mean():.3f}")
    print(f"correlation: max difference {drift:.3f}")
    print(f"log evidence: {ea} and {eb}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
