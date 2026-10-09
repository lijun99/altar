#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# check the kinematic gradient against jax: a port of the cuda slip history Mb, differentiated
# exactly, against {Kinematic.gradient}; needs jax (the cpu build will do); from
# models/seismic/examples, with the 9patch green's functions in place, run
#     python ../tests/kinematic_jax.py --config=kinematic.pfg \
#         --job.gpuprecision=float64 --job.chains=8


# externals
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import time
import types
import numpy
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
# the package
import altar
import altar.cuda


class SlipHistory:
    """
    A jax port of {cudaKinematic::calculateBigM}, kernel for kernel, for one sample: the
    arrival times seeded around the hypocenter, fast sweeping, their interpolation to the
    source points, and the triangular source time functions
    """

    def __init__(self, Nas, Ndd, Nmesh, dsp, Nt, Npt, dt, t0s, idx, it0=1.0e6, iterations=1,
                 radius=1.0, steepness=10.0):
        self.Nas, self.Ndd, self.Nmesh, self.dsp = Nas, Ndd, Nmesh, dsp
        self.Nt, self.Npt, self.dt, self.it0 = Nt, Npt, dt, it0
        self.radius, self.steepness = radius, steepness
        self.Np = Nas * Ndd
        self.Nasf, self.Nddf = (Nas + 2) * Nmesh, (Ndd + 2) * Nmesh
        self.dspf = dsp / Nmesh
        self.t0s = jnp.asarray(t0s)
        self.idx = numpy.asarray(idx)
        self.schedule = jnp.asarray(numpy.tile(self._sweeps(), (iterations, 1, 1)))
        self._interpolation()
        self._sources()


    def __call__(self, theta):
        """
        Mb[Nt][2][patches], flattened, at {theta}
        """
        θ = theta[self.idx]
        Np = self.Np
        vr = θ[3 * Np:4 * Np]
        T = jax.lax.scan(
            lambda T, step: (self._upwind(T, step, vr), None), self._seed(θ, vr), self.schedule)[0]
        # bilinear, in the kernel's order of operations
        f11, f21, f12, f22 = (T[c] for c in self.corners)
        xr, yr = self.xr, self.yr
        TI0 = f11 * (1.0 - xr) * (1.0 - yr) + f21 * xr * (1.0 - yr) \
            + f12 * (1.0 - xr) * yr + f22 * xr * yr
        # the triangles, at the start of each time interval
        hTr = (θ[2 * Np:3 * Np] / 2.0)[:, None, None]
        TI = TI0[self.subcells][:, None, :] + hTr
        t = (self.t0s[:, None] + self.dt * jnp.arange(self.Nt)[None, :])[:, :, None]
        inside = ~((t <= TI - hTr) | (t >= TI + hTr))
        triangle = (1.0 - jnp.abs(TI - t) / hTr) * (1.0 / hTr) / (self.Npt * self.Npt)
        c = jnp.where(inside, triangle, 0.0).sum(axis=-1)
        return jnp.stack([(c * θ[:Np, None]).T, (c * θ[Np:2 * Np, None]).T], axis=1).reshape(-1)


    def coverage(self):
        """
        The fraction of the mesh each of the four sweeps visits
        """
        steps = numpy.asarray(self.schedule).reshape(4, -1, 6, self.schedule.shape[-1])
        cells = self.Nasf * self.Nddf
        return [numpy.unique(s[:, 0][s[:, 0] < cells]).size / cells for s in steps]


    # implementation details
    def _patch(self, i, j):
        """
        The patch of the mesh point at strike {i}, dip {j}, clamped to the fault
        """
        Nmesh = self.Nmesh
        return numpy.clip(i // Nmesh - 1, 0, self.Nas - 1) * self.Ndd \
            + numpy.clip(j // Nmesh - 1, 0, self.Ndd - 1)


    def _seed(self, θ, vr):
        """
        The straight ray times from the hypocenter within {radius} mesh cells, rising steeply past
        it, capped at {it0}
        """
        dsp, dspf, Nddf = self.dsp, self.dspf, self.Nddf
        hs = θ[4 * self.Np] + dsp * 1.5
        hd = θ[4 * self.Np + 1] + dsp * 1.5
        i, j = numpy.divmod(numpy.arange(self.Nasf * Nddf), Nddf)
        ds = (i + 0.5) * dspf - hs
        dd = (j + 0.5) * dspf - hd
        distance = jnp.sqrt(ds * ds + dd * dd)
        far = jnp.maximum(0.0, distance - self.radius * dspf)
        T = jnp.minimum((distance + self.steepness * far * far / dspf) / vr[self._patch(i, j)], self.it0)
        # and the dump cell
        return jnp.append(T, self.it0)


    def _upwind(self, T, step, vr):
        """
        One half-step of a sweep: the upwind update of the cells on one diagonal
        """
        cell, xa, xb, ya, yb, patch = step
        f = 1.0 / vr[patch]
        fh = f * self.dspf
        ux = jnp.minimum(T[xa], T[xb])
        uy = jnp.minimum(T[ya], T[yb])
        d = ux - uy
        apart = jnp.abs(d) >= fh
        # the double where keeps the untaken sqrt out of the gradient
        root = jnp.sqrt(jnp.where(apart, 1.0, 2.0 * fh * fh - d * d))
        new = jnp.where(apart, jnp.minimum(ux, uy) + fh, (ux + uy + root) / 2.0)
        return T.at[cell].set(jnp.minimum(new, T[cell]))


    def _sweeps(self):
        """
        The cells each thread of {fastSweeping_batched} updates, half-step by half-step; idle
        threads update a dump cell past the end of the mesh
        """
        Nasf, Nddf = self.Nasf, self.Nddf
        Nf = max(Nasf, Nddf)
        ids = numpy.arange(Nf)
        half = Nf // 2
        dump = Nasf * Nddf
        starts = [(half - ids, ids - half, 1, 1), (ids + half, ids - half, -1, 1),
                  (half - ids + Nf - 1, ids + half, -1, -1), (ids - half, ids + half, 1, -1)]
        steps = []
        for nas, ndd, dnas, dndd in starts:
            for _ in range(Nf):
                for _ in range(2):
                    valid = (nas >= 0) & (nas < Nasf) & (ndd >= 0) & (ndd < Nddf)
                    i, j = numpy.where(valid, nas, 0), numpy.where(valid, ndd, 0)
                    neighbors = [i * Nddf + j,
                                 numpy.maximum(i - 1, 0) * Nddf + j,
                                 numpy.minimum(i + 1, Nasf - 1) * Nddf + j,
                                 i * Nddf + numpy.maximum(j - 1, 0),
                                 i * Nddf + numpy.minimum(j + 1, Nddf - 1)]
                    steps.append([numpy.where(valid, n, dump) for n in neighbors]
                                 + [self._patch(i, j)])
                    # each half-step moves along dip, then along strike
                    if len(steps) % 2:
                        ndd = ndd + dndd
                    else:
                        nas = nas + dnas
        return numpy.asarray(steps)


    def _interpolation(self):
        """
        The mesh corners and weights of each source point, as {interpolateT0_batched} has them
        """
        Nmesh, Npt, Nddf = self.Nmesh, self.Npt, self.Nddf
        offset = (Nmesh - 0.5) + 0.5 * Nmesh / Npt
        x = offset + numpy.arange(self.Ndd * Npt) * Nmesh / Npt
        y = offset + numpy.arange(self.Nas * Npt) * Nmesh / Npt
        # TI0[strike][dip], flattened
        X, Y = numpy.meshgrid(x, y)
        ix, iy = X.astype(int), Y.astype(int)
        self.xr = jnp.asarray((X - ix).ravel())
        self.yr = jnp.asarray((Y - iy).ravel())
        self.corners = [jnp.asarray((a + b * Nddf).ravel())
                        for a, b in [(ix, iy), (ix + 1, iy), (ix, iy + 1), (ix + 1, iy + 1)]]
        return


    def _sources(self):
        """
        The source points of each patch, dip first, as {castBigM_batched} sums them
        """
        Npt, Ndd = self.Npt, self.Ndd
        patches = numpy.arange(self.Np)
        dip, strike = patches % Ndd, patches // Ndd
        a, b = numpy.meshgrid(numpy.arange(Npt), numpy.arange(Npt), indexing="ij")
        self.subcells = jnp.asarray(
            (dip[:, None] * Npt + a.ravel()) + (strike[:, None] * Npt + b.ravel()) * Ndd * Npt)
        return


class Check(altar.shells.application, family="altar.applications.kinematicjax"):
    """
    Compare {Kinematic.gradient} with the exact gradient of a jax port of the model
    """

    model = altar.models.model(default="altar.models.seismic.kinematic")

    seed = altar.properties.int(default=1)
    seed.doc = "the seed of the random test points"

    output = altar.properties.str(default=None)
    output.doc = "an .npz file for the thetas, likelihoods and gradients"


    @altar.export
    def main(self, *args, **kwds):
        self.job.initialize(application=self)
        altar.backends.select_backend(application=self)
        self.rng.initialize()
        self.controller.initialize(application=self)
        m = self.model = self.model.initialize(application=self)
        n = m.samples
        Np = m.Nas * m.Ndd
        idx = numpy.asarray(m.gidx_map)
        theta = self.thetas(n=n, Np=Np, idx=idx)

        # altar: the slip histories, the likelihoods, the gradient
        θ = altar.cuda.matrix(source=theta, dtype=m.precision)
        # jax at the thetas altar sees, rounded to its precision
        theta = numpy.asarray(θ).astype(float)
        gMb = altar.cuda.matrix(shape=(n, m.NGbparameters), dtype=m.precision)
        m.cmodel.cast_mb(theta=θ.grid, mb=gMb.grid, batch=n)
        Mb = numpy.asarray(gMb).astype(float)
        llk = altar.cuda.vector(shape=n, dtype=m.precision)
        m.eval_data_likelihood(theta=θ, likelihood=llk, batch=n)
        llk = numpy.asarray(llk).astype(float) - m.dataobs.normalization
        step = types.SimpleNamespace(
            theta=θ,
            prior_gradient=altar.cuda.matrix(shape=theta.shape, dtype=m.precision).zero(),
            data_gradient=altar.cuda.matrix(shape=theta.shape, dtype=m.precision).zero())
        # only the data gradient is compared, so bounded priors don't matter
        m.checked_unbounded_priors = True
        m.gradient(controller=self.controller, step=step, batch=n)
        start = time.perf_counter()
        m.gradient(controller=self.controller, step=step, batch=n)
        altar_time = time.perf_counter() - start
        gradient = numpy.asarray(step.data_gradient).astype(float)

        # jax: the same, exactly
        history = SlipHistory(
            Nas=m.Nas, Ndd=m.Ndd, Nmesh=m.Nmesh, dsp=m.dsp, Nt=m.Nt, Npt=m.Npt, dt=m.dt,
            t0s=numpy.asarray(m.gt0s), idx=idx)
        GT = jnp.asarray(numpy.asarray(m.gGF).astype(float))
        data = jnp.asarray(numpy.asarray(m.dataobs.dataobs_batch)[0].astype(float))
        def loglikelihood(Mb):
            w = Mb @ GT - data
            return -0.5 * jnp.dot(w, w)
        Mb_jax = jax.jit(jax.vmap(history))
        value_and_grad = jax.jit(jax.vmap(jax.value_and_grad(lambda t: loglikelihood(history(t)))))
        Θ = jnp.asarray(theta)
        jax.block_until_ready(value_and_grad(Θ))
        start = time.perf_counter()
        llk_jax, gradient_jax = jax.block_until_ready(value_and_grad(Θ))
        jax_time = time.perf_counter() - start
        mb_jax = numpy.asarray(Mb_jax(Θ))

        # report; the tolerances leave room above what float64 and float32 reach
        tolerance = 1e-10 if m.precision == "float64" else 1e-3
        errors = {}
        print(f"fast sweeping coverage: {', '.join(f'{c:.1%}' for c in history.coverage())}")
        scale = numpy.abs(Mb).max()
        errors["Mb"] = numpy.abs(Mb - mb_jax).max() / scale
        errors["llk"] = (numpy.abs(llk - numpy.asarray(llk_jax)) / numpy.abs(llk)).max()
        print(f"Mb, max |cuda - jax| / max |Mb|: {errors['Mb']:.2e}")
        print(f"llk, max |cuda - jax| / |llk|: {errors['llk']:.2e}")
        groups = {"strike slips": idx[:Np], "dip slips": idx[Np:2 * Np],
                  "rise times": idx[2 * Np:3 * Np], "rupture velocities": idx[3 * Np:4 * Np],
                  "hypocenter": idx[4 * Np:]}
        print(f"the gradient in {m.precision}, max over samples of |altar - jax|_inf / |jax|_inf:")
        exact = numpy.asarray(gradient_jax)
        for name, cols in groups.items():
            norm = numpy.abs(exact[:, cols]).max(axis=1)
            error = (numpy.abs(gradient[:, cols] - exact[:, cols]).max(axis=1) / norm).max()
            errors[name] = error
            print(f"  {name:20s}{error:10.2e}")
        print("hypocenter gradient per sample: altar | jax")
        for k in range(n):
            print(f"  {k}: {gradient[k, groups['hypocenter']]} | {exact[k, groups['hypocenter']]}")
        print(f"gradient of {n} samples: altar (gpu) {altar_time:.3f}s, jax (cpu) {jax_time:.3f}s")

        if self.output:
            numpy.savez(self.output, theta=theta, llk=llk, llk_jax=numpy.asarray(llk_jax),
                        Mb=Mb, Mb_jax=mb_jax, gradient=gradient, gradient_jax=exact)
        # the verdict
        failed = {name: error for name, error in errors.items() if not error <= tolerance}
        for name, error in failed.items():
            print(f"FAIL: {name} differs from jax by {error:.2e}, above {tolerance:.0e}")
        print(f"against jax, within {tolerance:.0e}: {'FAIL' if failed else 'ok'}")
        return 1 if failed else 0


    def thetas(self, n, Np, idx):
        """
        The test points: the synthetic model, the same with its hypocenter off the mesh lines,
        then random ones
        """
        truth = numpy.loadtxt(os.path.join(str(self.model.case), "kinematicG_synthetic_theta.txt"))
        rng = numpy.random.default_rng(self.seed)
        points = [truth, truth.copy()]
        points[1][4 * Np:] += 0.123
        while len(points) < n:
            p = truth.copy()
            p[:Np] += rng.normal(0.0, 0.1, Np)
            p[Np:2 * Np] = numpy.abs(rng.normal(0.0, 0.2, Np))
            p[2 * Np:3 * Np] = rng.uniform(12.0, 28.0, Np)
            p[3 * Np:4 * Np] = rng.uniform(1.5, 5.0, Np)
            p[4 * Np:] += rng.uniform(-8.0, 8.0, 2)
            points.append(p)
        theta = numpy.empty((n, idx.size))
        theta[:, idx] = numpy.asarray(points[:n])
        return theta


# main
if __name__ == "__main__":
    raise SystemExit(Check(name="slipmodel").run())


# end of file
