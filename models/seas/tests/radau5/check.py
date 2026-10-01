#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2026 california institute of technology
# all rights reserved
#

"""
Check the radau5 method against exact and scipy solutions

    python check.py <run> <work dir>    # <run> is the binary built from run.cu

linear: s' = B z, z' = f A z, with A's eigenvalues from -1 to -1e5, against the exact solution
vanderpol: mu = 10, 100, 1000, against scipy's Radau at a tight tolerance
"""

import subprocess, sys, pathlib
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp


def write(d, sizes, **arrays):
    d.mkdir(parents=True, exist_ok=True)
    (d / "sizes.txt").write_text(" ".join(str(s) for s in sizes))
    for name, a in arrays.items():
        np.ascontiguousarray(a, dtype=float).tofile(d / f"{name}.bin")


def run(binary, problem, method, d, systems, neval, N):
    out = subprocess.run([binary, problem, method, str(d)], capture_output=True, text=True, check=True)
    y = np.fromfile(d / "yeval.bin").reshape(systems, neval, N)
    stats = np.loadtxt(d / "stats.txt", ndmin=2)
    return y, stats, out.stdout.strip()


def linear(binary, d):
    rng = np.random.default_rng(1)
    L, M, systems, neval, t1 = 10, 30, 16, 20, 1.0
    N = L + M
    # a non-normal, stiff A, and a stiffness factor for each system
    V = np.eye(M) + 0.3 * rng.standard_normal((M, M))
    A = -V @ np.diag(np.logspace(0, 5, M)) @ np.linalg.inv(V)
    B = rng.standard_normal((L, M))
    factor = np.logspace(0, 1, systems)
    y0 = np.concatenate([np.zeros(L), rng.standard_normal(M)])
    teval = np.linspace(t1 / neval, t1, neval)
    atol, rtol = 1e-8, 1e-6
    write(d, (systems, N, L, neval, t1, atol, rtol), A=A, B=B, factor=factor, y0=y0, teval=teval)

    # the exact solution
    exact = np.empty((systems, neval, N))
    for s, f in enumerate(factor):
        fA = f * A
        for k, t in enumerate(teval):
            E = expm(fA * t)
            z = E @ y0[L:]
            exact[s, k] = np.concatenate([B @ np.linalg.solve(fA, (E - np.eye(M)) @ y0[L:]), z])

    print(f"linear: {systems} systems, N = {N} ({L} inert), eigenvalues to -1e5 x [1, 10], t in [0, {t1}]")
    for method in ("radau5", "dopri5"):
        y, stats, msg = run(binary, "linear", method, d, systems, neval, N)
        scale = atol + rtol * np.abs(exact)
        err = np.abs(y - exact) / scale
        print(f"  {method}: {msg}; steps {int(stats[:, 0].min())}-{int(stats[:, 0].max())}, "
              f"rejected {int(stats[:, 1].sum())}, stiff {int(stats[:, 2].sum())}; "
              f"max error {err.max():.2f} tolerances (median {np.median(err):.3f})")


def vanderpol(binary, d):
    mu = np.array([10.0, 100.0, 1000.0])
    systems, N, neval, t1 = mu.size, 2, 20, 2000.0
    y0 = np.array([2.0, 0.0])
    teval = np.linspace(t1 / neval, t1, neval)
    atol, rtol = 1e-8, 1e-6
    write(d, (systems, N, 0, neval, t1, atol, rtol), mu=mu, y0=y0, teval=teval)

    print(f"vanderpol: mu = {mu.tolist()}, t in [0, {t1}]")
    y, stats, msg = run(binary, "vanderpol", "radau5", d, systems, neval, N)
    print(f"  radau5: {msg}")
    for s, m in enumerate(mu):
        ref = solve_ivp(lambda t, y: [y[1], m * (1 - y[0] ** 2) * y[1] - y[0]], (0, t1), y0,
                        method="Radau", t_eval=teval, rtol=1e-11, atol=1e-13).y.T
        steps = solve_ivp(lambda t, y: [y[1], m * (1 - y[0] ** 2) * y[1] - y[0]], (0, t1), y0,
                          method="Radau", rtol=rtol, atol=atol)
        # y0 has jumps of ~4 across the relaxation; measure the error against the solution scale
        err = np.abs(y[s] - ref).max(axis=0)
        print(f"    mu {m:6.0f}: steps {int(stats[s, 0])} (scipy {steps.t.size - 1}), "
              f"rejected {int(stats[s, 1])}; max |y - ref| {err[0]:.2e}, {err[1]:.2e}")


if __name__ == "__main__":
    binary, work = sys.argv[1], pathlib.Path(sys.argv[2])
    linear(binary, work / "linear")
    vanderpol(binary, work / "vanderpol")

# end of file
