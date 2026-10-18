# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import numpy
# get the package
import altar
# the samples and the misfit, as the forward check reads and measures them
from .Forward import Misfit, load_samples


# declaration
class Resolution(altar.panel(), family='altar.actions.resolution'):
    """
    How well the data resolve the parameters: the Fisher information of the data,
    F = J^T C_chi^{-1} J, with J the jacobian of the forward model at the posterior mean, and,
    with the conjugate prior of the model, N(m, C_m), the resolution matrix
    R = (F + C_m^{-1})^{-1} F and the effective number of parameters, tr(R)
    """


    # user configurable state
    theta = altar.properties.path(default="results/step_final.h5")
    theta.doc = "the posterior: an archived step (.h5 with ParameterSets), or a .txt/.h5 " \
                "file with one sample per row"

    dataset = altar.properties.str(default=None)
    dataset.doc = "the dataset holding {theta} in a plain .h5 file; default the first one"

    samples = altar.properties.int(default=None)
    samples.doc = "average F over this many posterior samples, for nonlinear models; " \
                  "default F at the posterior mean only"

    step = altar.properties.float(default=1e-3)
    step.doc = "the finite-difference step of each parameter, relative to its posterior " \
               "standard deviation"

    modes = altar.properties.int(default=10)
    modes.doc = "the number of the best-resolved parameter patterns to save"

    output = altar.properties.path(default="resolution.h5")
    output.doc = "the .h5 file for F, the resolution and the resolved patterns"


    # commands
    @altar.export(tip="measure how well the data resolve the parameters")
    def default(self, plexus, **kwds):
        """
        Compute F from each model's data, the resolution of each parameter and the leading
        resolved patterns, report them by parameter set and save them
        """
        model = plexus.model
        θ = load_samples(model=model, path=self.theta, dataset=self.dataset)
        mean = θ.mean(axis=0)
        sd = θ.std(axis=0) if θ.shape[0] > 1 else numpy.zeros_like(mean)
        # the finite-difference steps, from the posterior spread, or the size of the parameter
        scale = numpy.where(sd > 0, sd, numpy.maximum(numpy.abs(mean), 1.0))
        h = self.step * scale
        # the points where F is evaluated
        points = mean[None, :] if self.samples is None else θ[:self.samples]

        channel = plexus.info
        where = "the posterior mean" if self.samples is None else f"{points.shape[0]} samples"
        channel.line(f"resolution: F at {where} of '{self.theta}' -> '{self.output}'")
        channel.line(f"  finite differences, steps {self.step:g} x the posterior sd")

        # F, summed over the models of an ensemble, each on its own columns
        F = numpy.zeros((model.parameters, model.parameters))
        members = getattr(model, "models", None)
        for name, member, columns in (
                [(name, member, model.columns(name)) for name, member in members.items()]
                if members else [(None, model, numpy.arange(model.parameters))]):
            Fm = self.fisher(plexus=plexus, model=member, points=points[:, columns],
                             mean=mean[columns], h=h[columns])
            F[numpy.ix_(columns, columns)] += Fm
            if name is not None:
                channel.line(f"  {name}: {member.dataobs.observations} observations, "
                             f"tr F C_m = {numpy.trace(Fm):.4g}")

        # the conjugate prior, in which F is measured
        _, variance = model.conjugate_prior()
        s = numpy.sqrt(variance)
        # the prior-whitened F, A = S F S, S = C_m^{1/2}; R = S (I + A)^{-1} A S^{-1}
        A = s[:, None] * F * s[None, :]
        λ, V = numpy.linalg.eigh(A)
        λ, V = numpy.clip(λ[::-1], 0, None), V[:, ::-1]
        resolution = numpy.einsum("ik,k,ik->i", V, λ / (1 + λ), V)
        linearized = s * numpy.sqrt(numpy.einsum("ik,k,ik->i", V, 1 / (1 + λ), V))
        effective = float((λ / (1 + λ)).sum())
        patterns = (s[:, None] * V[:, :self.modes]).T
        patterns /= numpy.linalg.norm(patterns, axis=1, keepdims=True)

        self.report(model=model, resolution=resolution, linearized=linearized, sd=sd,
                    effective=effective, eigenvalues=λ, channel=channel)
        import h5py
        with h5py.File(str(self.output), "w") as h5:
            h5["theta/mean"], h5["theta/std"] = mean, sd
            h5["fisher"] = F
            h5["prior/variance"] = variance
            h5["resolution"] = resolution
            h5["effective_parameters"] = effective
            h5["eigenvalues"] = λ
            h5["patterns"] = patterns
            h5["linearized_std"] = linearized
        channel.log()
        return 0


    def fisher(self, plexus, model, points, mean, h):
        """
        F = J^T C_chi^{-1} J of {model}, averaged over {points}, with J by central differences
        """
        # C_p, if any, at the posterior mean, as in the forward check
        cp = getattr(model, "cp", None)
        if cp is not None:
            cp.apply(model=model, theta=mean)
        misfit = Misfit(dataobs=model.dataobs)
        parameters = mean.size
        F = numpy.zeros((parameters, parameters))
        for θ in points:
            # all the perturbed parameters in one batch, as a run's chains
            batch = numpy.vstack([θ + numpy.diag(h), θ - numpy.diag(h)])
            p = model.forward_problem(application=plexus, theta=batch)["data"]
            J = (p[:parameters] - p[parameters:]).T / (2 * h)
            W = misfit.whiten_jacobian(J)
            F += W.T @ W
        return F / points.shape[0]


    def report(self, model, resolution, linearized, sd, effective, eigenvalues, channel):
        """
        Summarize the resolution of each parameter set on {channel}
        """
        channel.line(f"  {'parameter set':16s}{'count':>6s}{'resolution':>12s}{'[min, max]':>16s}"
                     f"{'P_eff':>8s}{'sd ratio':>10s}")
        for name in model.psets_list:
            pset = model.psets[name]
            columns = slice(pset.offset, pset.offset + pset.count)
            r = resolution[columns]
            ratio = sd[columns] / linearized[columns]
            channel.line(f"  {name:16s}{pset.count:6d}{r.mean():12.3f}"
                         f"{f'[{r.min():.3f}, {r.max():.3f}]':>16s}{r.sum():8.2f}"
                         f"{numpy.median(ratio):10.3f}")
        constrained = int((eigenvalues > 1).sum())
        channel.line(f"  P_eff = tr R = {effective:.2f} of {model.parameters} parameters;"
                     f" {constrained} directions where the data outweigh the prior")
        channel.line("  (resolution: diag R, 1 resolved by the data, 0 by the prior alone;"
                     " sd ratio: the median of the sampled sd over the linearized sd)")
        return


# end of file
