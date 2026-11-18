# the slip a source point releases in all with sample, as a fraction of its slip, over the phases
# of its onset within an interval, by rise time; numpy only
import numpy


def rate(s, Tr, kind, peak):
    out = numpy.zeros_like(s)
    m = (s > 0) & (s < Tr)
    if kind == "cosine":
        out[m] = (1 - numpy.cos(2 * numpy.pi * s[m] / Tr)) / Tr
        return out
    up, down = m & (s < peak * Tr), m & (s >= peak * Tr)
    out[up] = 2 * s[up] / (peak * Tr * Tr)
    out[down] = 2 * (Tr - s[down]) / ((1 - peak) * Tr * Tr)
    return out


dt = 1.0
for kind, peak in [("triangle", 0.5), ("triangle", 0.25), ("triangle", 0.8), ("cosine", 0.5)]:
    for ratio in [0.5, 1, 1.5, 2, 2.5, 3, 5]:
        totals = [(rate(numpy.arange(-5, 40) * dt - phase, ratio * dt, kind, peak) * dt).sum()
                  for phase in numpy.linspace(0, dt, 401)]
        print(f"{kind} {peak:g}, Tr/dt {ratio:g}: {min(totals):.4f} to {max(totals):.4f}")
