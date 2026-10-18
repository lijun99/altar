# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import numpy
# get the package
import altar


# declaration
class Forward(altar.panel(), family='altar.actions.forward'):
    """
    Check a sampled model against its data: run the forward model on the posterior mean and on
    the posterior samples, and compare the predicted data, with its uncertainty band, to the
    observed data
    """


    # user configurable state
    theta = altar.properties.path(default="results/step_final.h5")
    theta.doc = "the parameters: an archived step (.h5 with ParameterSets), or a .txt/.h5 " \
                "file with one sample per row"

    dataset = altar.properties.str(default=None)
    dataset.doc = "the dataset holding {theta} in a plain .h5 file; default the first one"

    samples = altar.properties.int(default=None)
    samples.doc = "run at most this many posterior samples; default all of them"

    output = altar.properties.path(default="forward.h5")
    output.doc = "the .h5 file for the predictions and their statistics"

    reference = altar.properties.path(default=None)
    reference.doc = "another posterior to compare with, in the same forms as {theta}; " \
                    "its fit is measured with the same data and covariance"


    # commands
    @altar.export(tip="compare the data predicted by the posterior to the observed data")
    def default(self, plexus, **kwds):
        """
        Run the forward model on the posterior, save the predictions and report the fit; for an
        ensemble, for each of its models, on its own columns of theta and its own data
        """
        model = plexus.model
        θ = load_samples(model=model, path=self.theta, dataset=self.dataset)
        if self.samples is not None:
            θ = θ[:self.samples]
        reference = None
        if self.reference is not None:
            reference = load_samples(model=model, path=self.reference, dataset=self.dataset)
            if self.samples is not None:
                reference = reference[:self.samples]

        channel = plexus.info
        channel.line(f"forward: {θ.shape[0]} samples of '{self.theta}' -> '{self.output}'")
        if reference is not None:
            channel.line(f"reference: {reference.shape[0]} samples of '{self.reference}'")
        import h5py
        with h5py.File(str(self.output), "w") as h5:
            h5["theta/mean"], h5["theta/std"] = θ.mean(axis=0), θ.std(axis=0)
            members = getattr(model, "models", None)
            if members:
                for name, member in members.items():
                    channel.line(f"{name}:")
                    columns = model.columns(name)
                    self.check(plexus=plexus, model=member, theta=θ[:, columns],
                               reference=None if reference is None else reference[:, columns],
                               h5=h5.create_group(name), channel=channel, indent="  ")
            else:
                self.check(plexus=plexus, model=model, theta=θ, reference=reference, h5=h5,
                           channel=channel)
        channel.log()
        return 0


    def check(self, plexus, model, theta, h5, channel, reference=None, indent=""):
        """
        Run {model} on the mean of {theta} and on all of {theta}, compare with its observed data,
        write the results into {h5} and summarize them on {channel}; the same for {reference}
        """
        mean = theta.mean(axis=0)
        at_mean = model.forward_problem(application=plexus, theta=mean[None, :])
        ensemble = model.forward_problem(application=plexus, theta=theta)

        # C_p, if any, at my posterior mean; the reference is measured against the same C_chi
        cp = getattr(model, "cp", None)
        if cp is not None:
            cp.apply(model=model, theta=mean)

        observed = model.dataobs.observed()
        sigma = model.dataobs.sigma()
        sigma_chi = model.dataobs.sigma_chi()
        misfit = Misfit(dataobs=model.dataobs)
        predicted = ensemble["data"]
        residual = observed - at_mean["data"][0]
        band = numpy.sqrt(predicted.std(axis=0) ** 2 + sigma_chi ** 2)
        miss = numpy.abs(observed - predicted.mean(axis=0)) / band

        h5["data/observed"] = observed
        h5["data/sigma"], h5["data/sigma_chi"] = sigma, sigma_chi
        h5["data/residual"] = residual
        h5["data/whitened_residual"] = misfit.whiten(at_mean["data"])[0]
        for name, values in ensemble.items():
            h5[f"{name}/mean_model"] = at_mean[name][0]
            h5[f"{name}/mean"] = values.mean(axis=0)
            h5[f"{name}/std"] = values.std(axis=0)

        channel.line(f"{indent}{misfit.observations} observations, {misfit.kind}")
        channel.line(f"{indent}data sigma rms {numpy.sqrt((sigma ** 2).mean()):.6g},"
                     f" with C_p {numpy.sqrt((sigma_chi ** 2).mean()):.6g}")
        fit = misfit.measure(mean_model=at_mean["data"][0], samples=predicted)
        self.save(fit=fit, h5=h5)
        self.report(fit=fit, label=str(self.theta), channel=channel, indent=indent)
        channel.line(f"{indent}observations within the predictive band: "
                     f"{(miss <= 1).mean():.1%} at 1 sigma, {(miss <= 2).mean():.1%} at 2 sigma")

        if reference is not None:
            other = misfit.measure(
                mean_model=model.forward_problem(
                    application=plexus, theta=reference.mean(axis=0)[None, :])["data"][0],
                samples=model.forward_problem(application=plexus, theta=reference)["data"])
            self.save(fit=other, h5=h5.create_group("reference"))
            self.report(fit=other, label=str(self.reference), channel=channel, indent=indent)
            gain = fit["samples"]["loglikelihood"].mean() - other["samples"]["loglikelihood"].mean()
            channel.line(f"{indent}mean log L, this run - reference: {gain:+.6g}"
                         f" (chi^2 {-2 * gain + 0.0:+.6g})")
        return


    def report(self, fit, label, channel, indent=""):
        """
        Summarize the fit measures of a posterior on {channel}
        """
        channel.line(f"{indent}{label}:")
        channel.line(f"{indent}  {'':22s}{'chi^2/N':>10s}{'VR':>10s}{'rms':>12s}{'log L':>14s}")
        for key, title in [("mean_model", "mean model"), ("predictive_mean", "predictive mean"),
                           ("best_sample", "best-fitting sample")]:
            m = fit[key]
            channel.line(f"{indent}  {title:22s}{m['chi2'] / fit['observations']:10.4g}"
                         f"{m['vr']:10.4f}{m['rms']:12.4g}{m['loglikelihood']:14.6g}")
        s = fit["samples"]
        llk = s["loglikelihood"]
        channel.line(f"{indent}  samples: log L {llk.mean():.6g} +- {llk.std():.3g},"
                     f" chi^2/N median {numpy.median(s['chi2']) / fit['observations']:.4g};"
                     f" 2 var(log L) = {2 * llk.var():.3g} (the effective number of parameters,"
                     f" for a gaussian posterior)")
        return


    def save(self, fit, h5):
        """
        Write the fit measures of a posterior into {h5}, under fit/
        """
        h5["fit/observations"] = fit["observations"]
        for key in ["mean_model", "predictive_mean", "best_sample"]:
            for name, value in fit[key].items():
                h5[f"fit/{key}/{name}"] = value
        h5["fit/samples/chi2"] = fit["samples"]["chi2"]
        h5["fit/samples/loglikelihood"] = fit["samples"]["loglikelihood"]
        return


def load_samples(model, path, dataset=None):
    """
    Read the samples at {path}, as a (samples x parameters) numpy array in physical space: an
    archived step, a .txt file with one sample per row, or {dataset} of a plain .h5 file
    """
    import h5py
    path = str(path)
    if path.endswith(".txt"):
        θ = numpy.loadtxt(path, dtype=float)
    else:
        with h5py.File(path, "r") as h5:
            if "ParameterSets" in h5:
                θ = archived(sets=h5["ParameterSets"], model=model, path=path)
            else:
                θ = numpy.asarray(h5[dataset or list(h5.keys())[0]], dtype=float)
    return θ.reshape(-1, model.parameters)


def archived(sets, model, path):
    """
    Assemble theta from an archived step, one parameter set at a time in {psets_list} order,
    preferring its physical-space samples when it was reparameterized
    """
    columns = []
    for name in model.psets_list:
        for key in (f"{name}_physical", f"{name}_sampling", name):
            if key in sets:
                columns.append(numpy.asarray(sets[key], dtype=float))
                break
        else:
            raise KeyError(f"no samples for parameter set '{name}' in '{path}'")
    return numpy.hstack(columns)


class Misfit:
    """
    The misfit of predicted data against the observations, under the covariance in effect,
    C_d or C_chi = C_d + C_p, whitened in double precision
    """

    def __init__(self, dataobs):
        self.observed = numpy.asarray(dataobs.observed(), dtype=float)
        self.mask = dataobs.mask
        covariance = dataobs.covariance()
        size = self.observed.size
        self.observations = size if self.mask is None else int(self.mask.sum())
        if numpy.ndim(covariance) == 0:
            # a common variance: the whitening is a scale
            variance = float(covariance)
            self.factor = None
            self.scale = 1 / numpy.sqrt(variance)
            self.normalization = -0.5 * self.observations * numpy.log(2 * numpy.pi * variance)
            self.kind = f"C = {numpy.sqrt(variance):.6g}^2 I"
        else:
            # L, the lower cholesky factor of C^{-1} = L L^T, as the likelihood uses
            C = numpy.asarray(covariance, dtype=float)
            self.factor = numpy.linalg.cholesky(numpy.linalg.inv(C))
            self.scale = None
            _, logdet = numpy.linalg.slogdet(C)
            self.normalization = -0.5 * (size * numpy.log(2 * numpy.pi) + logdet)
            self.kind = "full covariance"
        return


    def whiten(self, predicted):
        """
        The whitened residuals, L^T (d - p), of the rows of {predicted}
        """
        r = self.observed - numpy.atleast_2d(predicted)
        if self.mask is not None:
            r = r * self.mask
        return r * self.scale if self.factor is None else r @ self.factor


    def whiten_jacobian(self, J):
        """
        L^T J, for the (observations x parameters) jacobian {J} of the predictions
        """
        if self.mask is not None:
            J = J * self.mask[:, None]
        return J * self.scale if self.factor is None else self.factor.T @ J


    def measure(self, mean_model, samples):
        """
        The fit of the mean model, of the predictive mean and of the best-fitting sample, and the
        chi^2 and the data log likelihood of each sample
        """
        chi2 = (self.whiten(samples) ** 2).sum(axis=1)
        best = int(numpy.argmin(chi2))
        fit = {"observations": self.observations,
               "samples": {"chi2": chi2, "loglikelihood": self.normalization - 0.5 * chi2}}
        for key, p in [("mean_model", mean_model), ("predictive_mean", samples.mean(axis=0)),
                       ("best_sample", samples[best])]:
            fit[key] = self.summary(predicted=p)
        fit["best_sample"]["index"] = best
        return fit


    def summary(self, predicted):
        """
        chi^2, the variance reduction, the rms residual and the log likelihood of {predicted}
        """
        d = self.observed if self.mask is None else self.observed * self.mask
        r = d - predicted if self.mask is None else (d - predicted) * self.mask
        chi2 = float((self.whiten(predicted) ** 2).sum())
        return {"chi2": chi2,
                "vr": 1 - float((r ** 2).sum() / (d ** 2).sum()),
                "rms": float(numpy.sqrt((r ** 2).sum() / self.observations)),
                "loglikelihood": self.normalization - 0.5 * chi2}


# end of file
