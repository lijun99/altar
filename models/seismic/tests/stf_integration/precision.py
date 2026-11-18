# float32 against float64 of each integration mode on rectangles, by the ratio of rise time to dt
import numpy
from altar.models.seismic.ext import seismic
rng = numpy.random.default_rng(1)
Nas, Ndd, B, P = 3, 2, 4, 6
for ratio in [0.5, 2, 10, 30, 100]:
    dt = 0.25
    Tr = ratio * dt
    Nt = int((20 + Tr) / dt) + 4
    row = []
    for mode in range(3):
        out = {}
        for prec in ["float64", "float32"]:
            k = seismic.Kinematic(Nas=Nas, Ndd=Ndd, Nmesh=6, dsp=10.0, Nt=Nt, Npt=2, dt=dt, t0s=numpy.zeros(P),
                                  idx_map=numpy.arange(4 * P + 2, dtype="int64"), precision=prec)
            k.set_stf(kind=0, peak=0.5, mode=mode)
            r = numpy.random.default_rng(2)
            t = numpy.column_stack([r.normal(0, .3, (B, P)), r.uniform(1, 3, (B, P)), Tr * r.uniform(0.9, 1.1, (B, P)),
                                    r.uniform(2, 3.5, (B, P)), r.uniform(5, 25, B), r.uniform(5, 15, B)]).astype(prec)
            Mb = numpy.zeros((B, 2 * P * Nt), dtype=prec)
            k.cast_mb(theta=t, mb=Mb, batch=B)
            out[prec] = Mb.astype(float)
        row.append(numpy.abs(out["float32"] - out["float64"]).max() / numpy.abs(out["float64"]).max())
    print(f"Tr/dt {ratio:5g}: float32 vs float64, sample {row[0]:.1e}  step {row[1]:.1e}  bin {row[2]:.1e}")
