#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the gradient of a function of Mb, through the source time functions, the interpolation and the
# eikonal adjoint, against central differences
# run from this directory: python check_slip.py
import numpy
from mesh import rectangle_fault
from slip import SlipHistory, rectangle_sources

Nas, Ndd, dsp, Npt, Nt = 3, 3, 20.0, 2, 90
Np = Nas * Ndd
mesh = rectangle_fault(Nas, Ndd, dsp, 10)
points, patch = rectangle_sources(Nas, Ndd, dsp, Npt)
sh = SlipHistory(mesh, points, patch, Np, Nt, 1.0, numpy.zeros(Np), Npt * Npt)
rng = numpy.random.default_rng(3)
worst = 0
for trial in range(5):
    θ = numpy.r_[rng.normal(0, 0.5, Np), rng.uniform(0, 5, Np), rng.uniform(10, 30, Np),
                 rng.uniform(1.5, 5, Np), rng.uniform(5, 35, 2)]
    r = rng.standard_normal(Nt * 2 * Np)
    sh(θ)
    g = sh.adjoint(r)
    f = lambda x: r @ sh(x)
    ε = 1e-6
    fd = numpy.array([(f(θ + ε * e) - f(θ - ε * e)) / (2 * ε) for e in numpy.eye(θ.size)])
    sets = {"slips": slice(0, 2 * Np), "risetime": slice(2 * Np, 3 * Np),
            "velocity": slice(3 * Np, 4 * Np), "hypocenter": slice(4 * Np, 4 * Np + 2)}
    line = []
    for name, s in sets.items():
        rel = numpy.abs(g[s] - fd[s]).max() / numpy.abs(fd[s]).max()
        worst = max(worst, rel)
        line.append(f"{name} {rel:.1e}")
    print(f"trial {trial}: max rel diff, " + ", ".join(line))
print(f"worst {worst:.1e}")


# end of file
