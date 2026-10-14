#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# Mb from the triangle mesh against the cuda kinematic model, same theta, same spacing; run from
# the seismic examples, with the 9patch inputs and a gpu:
#     python ../tests/trimesh/compare_cuda.py <Nmesh>
import os, sys
n = int(sys.argv.pop(1))
sys.argv += ["--config=kinematic.pfg", "--job.gpus=1", "--job.chains=64", f"--slipmodel.model.Nmesh={n}"]
import numpy
import altar, altar.models.seismic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh import rectangle_fault
from slip import SlipHistory, rectangle_sources

app = altar.shells.altar(name="slipmodel").initialize()
model = app.model
Nas, Ndd, dsp, Npt, Nt, dt = model.Nas, model.Ndd, model.dsp, model.Npt, model.Nt, model.dt
Np = Nas * Ndd
points, patch = rectangle_sources(Nas, Ndd, dsp, Npt)
sh = SlipHistory(rectangle_fault(Nas, Ndd, dsp, n), points, patch, Np, Nt, dt,
                 numpy.asarray(model.t0s, dtype=float), Npt * Npt)
rng = numpy.random.default_rng(4)
rows = []
for k in range(8):
    θ = numpy.r_[rng.normal(0, 0.5, Np), rng.uniform(0, 5, Np), rng.uniform(10, 30, Np),
                 rng.uniform(1.5, 5, Np), rng.uniform(5, 35, 2)]
    rows.append(θ)
θ = numpy.asarray(rows)
cuda = model.forward_problem(application=app, theta=θ)["Mb"]
tri = numpy.asarray([sh(t) for t in θ])
rel = numpy.linalg.norm(tri - cuda, axis=1) / numpy.linalg.norm(cuda, axis=1)
print(f"Nmesh {n} (spacing {dsp / n:.2f} km): |Mb tri - Mb cuda| / |Mb cuda| over 8 models:"
      f" median {numpy.median(rel):.4f}, max {rel.max():.4f}")

# constant velocity: both against the exact straight-ray arrival times
rows = []
for k in range(8):
    rows.append(numpy.r_[rng.normal(0, 0.5, Np), rng.uniform(0, 5, Np), rng.uniform(10, 30, Np),
                         numpy.full(Np, rng.uniform(1.5, 5)), rng.uniform(5, 35, 2)])
θ = numpy.asarray(rows)
cuda = model.forward_problem(application=app, theta=θ)["Mb"]
tri = numpy.asarray([sh(t) for t in θ])
exact = []
for t in θ:
    TI0 = numpy.sqrt(((points - t[4 * Np:]) ** 2).sum(1)) / t[3 * Np]
    c, _ = sh.stf(TI0=TI0, h=t[2 * Np:3 * Np] / 2.0)
    exact.append(numpy.stack([(c * t[:Np, None]).T, (c * t[Np:2 * Np, None]).T], axis=1).ravel())
exact = numpy.asarray(exact)
norm = numpy.linalg.norm(exact, axis=1)
e_c = numpy.linalg.norm(cuda - exact, axis=1) / norm
e_t = numpy.linalg.norm(tri - exact, axis=1) / norm
print(f"   constant velocity, error against exact: cuda median {numpy.median(e_c):.4f} max {e_c.max():.4f};"
      f" triangles median {numpy.median(e_t):.4f} max {e_t.max():.4f}")


# end of file
