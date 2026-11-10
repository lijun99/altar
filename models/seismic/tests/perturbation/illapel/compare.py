#!/usr/bin/env python3
"""
Compare the final samples of the Illapel runs with the exact gaussian posterior of static_gauss.pfg
    python compare.py <name> ...
"""
import math, sys, pathlib
import h5py, numpy

INPUT = pathlib.Path("/baghdad-nobak/ljzhu/kinematic/Illapel.orig/Illapel_input")
RESULTS = pathlib.Path("/baghdad-nobak/ljzhu/kinematic/Illapel.orig/altar2/results")
SETS = [("strikeslip", 170, 1.0), ("dipslip", 170, 5.0), ("ramp", 2, 5.0)]

def exact():
    G = numpy.asarray(h5py.File(INPUT / "static.gf.h5")["static.gf"], dtype=float)
    d = numpy.asarray(h5py.File(INPUT / "static.data.h5")["static.data"], dtype=float)
    C = numpy.asarray(h5py.File(INPUT / "static.Cxi.h5")["static.Cxi"], dtype=float)
    sigma = numpy.concatenate([numpy.full(n, s) for _, n, s in SETS])
    L = numpy.linalg.cholesky(C)
    Gw, dw = numpy.linalg.solve(L, G), numpy.linalg.solve(L, d)
    A = Gw.T @ Gw + numpy.diag(1 / sigma ** 2)
    S = numpy.linalg.inv(A); S = (S + S.T) / 2
    m = S @ (Gw.T @ dw)
    # log N(d; 0, C + G Σp G^T)
    K = C + (G * sigma ** 2) @ G.T
    evidence = -0.5 * (d @ numpy.linalg.solve(K, d) + numpy.linalg.slogdet(K)[1] + d.size * math.log(2 * math.pi))
    return m, S, evidence

def samples(name):
    with h5py.File(RESULTS / name / "step_final.h5") as h5:
        sets = h5["ParameterSets"]
        pick = lambda s: sets[f"{s}_physical" if f"{s}_physical" in sets else f"{s}_sampling"]
        theta = numpy.hstack([numpy.asarray(pick(s), dtype=float) for s, _, _ in SETS])
        e = h5["Annealer"].get("log_evidence")
        return theta, None if e is None else float(numpy.asarray(e))

m, S, evidence = exact()
sd = numpy.sqrt(numpy.diag(S))
print(f"exact: {m.size} parameters, log evidence {evidence:.2f}; ramp {m[-2:].round(3)} ± {sd[-2:].round(3)}")
print(f"{'run':24s} {'samples':>7s} {'mean |Δ|/sd':>16s} {'sd ratio':>18s} {'corr':>6s} {'log evidence':>12s}")
for name in sys.argv[1:]:
    theta, e = samples(name)
    shift = numpy.abs(theta.mean(axis=0) - m) / sd
    ratio = theta.std(axis=0) / sd
    R = S / numpy.outer(sd, sd)
    drift = numpy.abs(numpy.corrcoef(theta, rowvar=False) - R).max()
    es = "" if e is None else f"{e:.2f}"
    print(f"{name:24s} {theta.shape[0]:7d} {shift.max():7.3f} rms {numpy.sqrt((shift**2).mean()):.3f} "
          f"[{ratio.min():.3f}, {ratio.max():.3f}] med {numpy.median(ratio):.3f} {drift:6.3f} {es:>12s}")
