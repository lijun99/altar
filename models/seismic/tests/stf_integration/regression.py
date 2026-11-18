# run once with a build of 8e5b872, before integration, and once with one after, then compare:
#   python regression.py old.npz <worktree>; python regression.py new.npz <worktree>
# sample mode against the build before exact integration: Mb and the gradient of r.Mb, bit for bit,
# on triangles (SlipHistory's compiled model) and rectangles (seismic.Kinematic directly)
import sys, numpy
sys.path.insert(0, sys.argv[2] + "/models/seismic/tests"); sys.path.insert(0, sys.argv[2] + "/models/seismic/tests/trimesh")
from kinematic_mesh import fault
from altar.models.seismic.history import SlipHistory
from altar.models.seismic.ext import seismic
rng = numpy.random.default_rng(3)
out = {}
xyz, uv, faces, sources, owners = fault("planar")
Np, Nt, B = len(faces), 40, 5
for stf, peak in [("triangle", 0.25), ("cosine", 0.5)]:
    h = SlipHistory(xyz, faces, uv, sources, owners, Nt=Nt, dt=0.5, Nmesh=6, stf=stf, peak=peak)
    θ = numpy.column_stack([rng.normal(0, .3, (B, Np)), rng.uniform(1, 3, (B, Np)), rng.uniform(0.2, 6, (B, Np)),
                            rng.uniform(2, 3.5, (B, Np)), rng.uniform(12, 18, (B, 2))])
    Mb = numpy.zeros((B, 2 * Np * Nt)); g = numpy.zeros_like(θ)
    h.model.cast_mb(theta=θ, mb=Mb, batch=B)
    h.model.gradient(theta=θ, dmb=rng.standard_normal(Mb.shape), gradient=g, batch=B)
    out[f"tri-{stf}-Mb"], out[f"tri-{stf}-g"] = Mb, g
    Nas, Ndd = 3, 2; P = Nas * Ndd
    for prec in ["float64", "float32"]:
        k = seismic.Kinematic(Nas=Nas, Ndd=Ndd, Nmesh=6, dsp=10.0, Nt=Nt, Npt=2, dt=0.5, t0s=numpy.zeros(P),
                              idx_map=numpy.arange(4 * P + 2, dtype="int64"), precision=prec)
        k.set_stf(kind={"triangle": 0, "cosine": 1}[stf], peak=peak)
        t = numpy.column_stack([rng.normal(0, .3, (B, P)), rng.uniform(1, 3, (B, P)), rng.uniform(0.2, 6, (B, P)),
                                rng.uniform(2, 3.5, (B, P)), rng.uniform(5, 25, B), rng.uniform(5, 15, B)]).astype(prec)
        Mb = numpy.zeros((B, 2 * P * Nt), dtype=prec); g = numpy.zeros_like(t)
        k.cast_mb(theta=t, mb=Mb, batch=B)
        k.gradient(theta=t, dmb=rng.standard_normal(Mb.shape).astype(prec), gradient=g, batch=B)
        out[f"rect-{stf}-{prec}-Mb"], out[f"rect-{stf}-{prec}-g"] = Mb, g
numpy.savez(sys.argv[1], **out)
