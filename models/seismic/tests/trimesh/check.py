#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# checks of the triangle-mesh eikonal solver and its adjoint, on the 9-patch fault
# run from this directory: python check.py
import time
import numpy
from mesh import Eikonal, rectangle_fault

Nas, Ndd, dsp = 3, 3, 20.0
rng = numpy.random.default_rng(0)

# 1. constant velocity: the arrival time converges to the straight distance over the velocity
print("1. constant velocity, error against |x - h| / v (the seed covers 1 spacing around h)")
for n in [5, 10, 20, 40]:
    mesh = rectangle_fault(Nas, Ndd, dsp, n)
    eik = Eikonal(mesh)
    v, h = numpy.full(Nas * Ndd, 3.0), numpy.array([13.0, 27.0])
    start = time.perf_counter()
    T = eik.solve(v, h)
    exact = numpy.sqrt(((mesh.uv - h) ** 2).sum(1)) / 3.0
    err = numpy.abs(T - exact)
    print(f"   n {n:3d}: {mesh.vertices.shape[0]:6d} vertices, {eik.sweeps:3d} iterations,"
          f" max error {err.max():.4f} s, mean {err.mean():.4f} s, {time.perf_counter() - start:.2f} s")

# 2. the adjoint against central differences, for a random function of the arrival times
print("2. adjoint against central differences, varied velocities")
mesh = rectangle_fault(Nas, Ndd, dsp, 10)
eik = Eikonal(mesh)
v = rng.uniform(2.0, 5.0, Nas * Ndd)
h = numpy.array([17.3, 24.1])
gT = rng.standard_normal(mesh.vertices.shape[0])
eik.solve(v, h)
g_v, g_h = eik.adjoint(gT)
print(f"   order violations in the backward pass: {eik.violations}")
f = lambda v, h: gT @ Eikonal(mesh).solve(v, h)
ε = 1e-6
fd_v = numpy.array([(f(v + ε * e, h) - f(v - ε * e, h)) / (2 * ε) for e in numpy.eye(v.size)])
fd_h = numpy.array([(f(v, h + ε * e) - f(v, h - ε * e)) / (2 * ε) for e in numpy.eye(2)])
for name, g, fd in [("velocity", g_v, fd_v), ("hypocenter", g_h, fd_h)]:
    rel = numpy.abs(g - fd).max() / numpy.abs(fd).max()
    print(f"   {name:10s} adjoint {numpy.array2string(g[:3], precision=5)}"
          f" fd {numpy.array2string(fd[:3], precision=5)}  max rel diff {rel:.2e}")

# 3. many random points: the worst relative difference
print("3. 20 random models")
worst = 0.0
for trial in range(20):
    v = rng.uniform(1.0, 6.0, Nas * Ndd)
    h = rng.uniform(-5, 45, 2)
    gT = rng.standard_normal(mesh.vertices.shape[0])
    eik = Eikonal(mesh)
    eik.solve(v, h)
    g_v, g_h = eik.adjoint(gT)
    f = lambda v, h: gT @ Eikonal(mesh).solve(v, h)
    fd = numpy.array([(f(v + ε * e, h) - f(v - ε * e, h)) / (2 * ε) for e in numpy.eye(v.size)]
                     + [(f(v, h + ε * e) - f(v, h - ε * e)) / (2 * ε) for e in numpy.eye(2)])
    g = numpy.r_[g_v, g_h]
    rel = numpy.abs(g - fd).max() / numpy.abs(fd).max()
    worst = max(worst, rel)
    if rel > 1e-4:
        print(f"   trial {trial}: max rel diff {rel:.2e}, violations {eik.violations}")
print(f"   worst max rel diff {worst:.2e}")

# 4. a fault whose dip changes from 20 to 70 degrees at 23.3 km down dip, through the triangles:
# developable, so the arrival times are the straight distances in the unfolded surface coordinates,
# up to the mesh's cut across the bend
print("4. a bent fault, dip 20 then 70 degrees, the bend through the triangles, against |uv - h| / v")
def bent(uv, kink=23.3, dips=(20.0, 70.0)):
    u, w = uv[:, 0], uv[:, 1] + 0.5 * dsp
    d1, d2 = numpy.radians(dips)
    w1, w2 = numpy.minimum(w, kink), numpy.maximum(w - kink, 0.0)
    return numpy.column_stack([u, w1 * numpy.cos(d1) + w2 * numpy.cos(d2),
                               -(w1 * numpy.sin(d1) + w2 * numpy.sin(d2))])
for n in [10, 20, 40]:
    mesh = rectangle_fault(Nas, Ndd, dsp, n, shape=bent)
    eik = Eikonal(mesh)
    h = numpy.array([13.0, 27.0])
    T = eik.solve(numpy.full(Nas * Ndd, 3.0), h)
    exact = numpy.sqrt(((mesh.uv - h) ** 2).sum(1)) / 3.0
    err = numpy.abs(T - exact)
    print(f"   n {n:3d}: max error {err.max():.4f} s, mean {err.mean():.4f} s")
mesh = rectangle_fault(Nas, Ndd, dsp, 10, shape=bent)
worst = 0.0
for trial in range(10):
    v = rng.uniform(1.0, 6.0, Nas * Ndd)
    h = rng.uniform(-5, 45, 2)
    gT = rng.standard_normal(mesh.vertices.shape[0])
    eik = Eikonal(mesh)
    eik.solve(v, h)
    f = lambda v, h: gT @ Eikonal(mesh).solve(v, h)
    fd = numpy.array([(f(v + ε * e, h) - f(v - ε * e, h)) / (2 * ε) for e in numpy.eye(v.size)]
                     + [(f(v, h + ε * e) - f(v, h - ε * e)) / (2 * ε) for e in numpy.eye(2)])
    g = numpy.concatenate(eik.adjoint(gT))
    worst = max(worst, numpy.abs(g - fd).max() / numpy.abs(fd).max())
print(f"   adjoint against central differences, 10 random models: worst max rel diff {worst:.2e}")


# end of file
