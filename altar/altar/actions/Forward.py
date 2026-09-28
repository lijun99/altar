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


    # commands
    @altar.export(tip="compare the data predicted by the posterior to the observed data")
    def default(self, plexus, **kwds):
        """
        Run the forward model on the posterior, save the predictions and report the fit; for an
        ensemble, for each of its models, on its own columns of theta and its own data
        """
        model = plexus.model
        θ = self.load_theta(model=model)
        if self.samples is not None:
            θ = θ[:self.samples]

        channel = plexus.info
        channel.line(f"forward: {θ.shape[0]} samples of '{self.theta}' -> '{self.output}'")
        import h5py
        with h5py.File(str(self.output), "w") as h5:
            h5["theta/mean"], h5["theta/std"] = θ.mean(axis=0), θ.std(axis=0)
            members = getattr(model, "models", None)
            if members:
                for name, member in members.items():
                    channel.line(f"{name}:")
                    self.check(plexus=plexus, model=member, theta=θ[:, model.columns(name)],
                               h5=h5.create_group(name), channel=channel, indent="  ")
            else:
                self.check(plexus=plexus, model=model, theta=θ, h5=h5, channel=channel)
        channel.log()
        return 0


    def check(self, plexus, model, theta, h5, channel, indent=""):
        """
        Run {model} on the mean of {theta} and on all of {theta}, compare with its observed data,
        write the results into {h5} and summarize them on {channel}
        """
        mean = theta.mean(axis=0)
        at_mean = model.forward_problem(application=plexus, theta=mean[None, :])
        ensemble = model.forward_problem(application=plexus, theta=theta)

        # the model uncertainty at the posterior mean, if the model has one
        cp = getattr(model, "cp", None)
        if cp is not None:
            cp.apply(model=model, theta=mean)

        # the data, and how well the predictions fit it, given the data uncertainty and the
        # model uncertainty C_p, if any
        observed = model.dataobs.observed()
        sigma = model.dataobs.sigma()
        sigma_chi = model.dataobs.sigma_chi()
        predicted = ensemble["data"]
        residual = observed - at_mean["data"][0]
        band = numpy.sqrt(predicted.std(axis=0) ** 2 + sigma_chi ** 2)
        miss = numpy.abs(observed - predicted.mean(axis=0)) / band

        h5["data/observed"] = observed
        h5["data/sigma"], h5["data/sigma_chi"] = sigma, sigma_chi
        h5["data/residual"] = residual
        for name, values in ensemble.items():
            h5[f"{name}/mean_model"] = at_mean[name][0]
            h5[f"{name}/mean"] = values.mean(axis=0)
            h5[f"{name}/std"] = values.std(axis=0)

        channel.line(f"{indent}mean model rms residual: {numpy.sqrt((residual ** 2).mean()):.6g}"
                     f" (data sigma rms {numpy.sqrt((sigma ** 2).mean()):.6g},"
                     f" with C_p {numpy.sqrt((sigma_chi ** 2).mean()):.6g})")
        channel.line(f"{indent}mean model chi^2/N, diagonal: {((residual / sigma_chi) ** 2).mean():.4g}")
        channel.line(f"{indent}observations within the predictive band: "
                     f"{(miss <= 1).mean():.1%} at 1 sigma, {(miss <= 2).mean():.1%} at 2 sigma")
        return


    # implementation details
    def load_theta(self, model):
        """
        Read the samples to run, as a (samples x parameters) numpy array in physical space
        """
        import h5py
        path = str(self.theta)
        if path.endswith(".txt"):
            θ = numpy.loadtxt(path, dtype=float)
        else:
            with h5py.File(path, "r") as h5:
                if "ParameterSets" in h5:
                    θ = self.archived(sets=h5["ParameterSets"], model=model)
                else:
                    θ = numpy.asarray(h5[self.dataset or list(h5.keys())[0]], dtype=float)
        return θ.reshape(-1, model.parameters)


    def archived(self, sets, model):
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
                raise KeyError(f"no samples for parameter set '{name}' in '{self.theta}'")
        return numpy.hstack(columns)


# end of file
