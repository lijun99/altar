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
# the samples, as the forward check reads them
from .Forward import load_samples


# declaration
class Recover(altar.panel(), family='altar.actions.recover'):
    """
    Compare a posterior with the true model of a recovery test, e.g. from {synthetic}: how far
    the posterior mean is from the truth, in posterior standard deviations, and how often the
    truth lies within the posterior's 68% and 95% intervals
    """


    # user configurable state
    theta = altar.properties.path(default="results/step_final.h5")
    theta.doc = "the posterior: an archived step (.h5 with ParameterSets), or a .txt/.h5 " \
                "file with one sample per row"

    dataset = altar.properties.str(default=None)
    dataset.doc = "the dataset holding {theta} in a plain .h5 file; default the first one"

    truth = altar.properties.path(default="synthetic/truth.txt")
    truth.doc = "the true model, a .txt file with one row"

    output = altar.properties.path(default="recover.h5")
    output.doc = "the .h5 file for the comparison of each parameter"


    # commands
    @altar.export(tip="compare a posterior with a true model")
    def default(self, plexus, **kwds):
        """
        Compare the posterior with the truth, parameter by parameter, and summarize by
        parameter set
        """
        model = plexus.model
        θ = load_samples(model=model, path=self.theta, dataset=self.dataset)
        truth = numpy.loadtxt(str(self.truth), dtype=float).reshape(-1)
        if truth.size != model.parameters:
            raise ValueError(f"'{self.truth}' has {truth.size} values, for "
                             f"{model.parameters} parameters")
        mean, sd = θ.mean(axis=0), θ.std(axis=0)
        z = (truth - mean) / numpy.where(sd > 0, sd, numpy.inf)
        # the central intervals of the samples
        inside = {}
        for level in (68, 95):
            low, high = numpy.percentile(θ, [50 - level / 2, 50 + level / 2], axis=0)
            inside[level] = (low <= truth) & (truth <= high)

        channel = plexus.info
        channel.line(f"recover: {θ.shape[0]} samples of '{self.theta}' against '{self.truth}'")
        channel.line(f"  {'parameter set':16s}{'count':>6s}{'rms z':>8s}{'max |z|':>9s}"
                     f"{'in 68%':>8s}{'in 95%':>8s}{'corr':>7s}")
        for name in model.psets_list:
            pset = model.psets[name]
            c = slice(pset.offset, pset.offset + pset.count)
            # the correlation of mean and truth, where the truth varies
            corr = (numpy.corrcoef(mean[c], truth[c])[0, 1]
                    if pset.count > 1 and truth[c].std() > 0 and mean[c].std() > 0 else numpy.nan)
            channel.line(f"  {name:16s}{pset.count:6d}{numpy.sqrt((z[c] ** 2).mean()):8.2f}"
                         f"{numpy.abs(z[c]).max():9.2f}{inside[68][c].mean():8.0%}"
                         f"{inside[95][c].mean():8.0%}{corr:7.3f}")
        channel.line(f"  all: truth within the 68% interval for {inside[68].mean():.0%} of the"
                     f" parameters, within the 95% one for {inside[95].mean():.0%}")
        channel.line("  (z: (truth - mean) / sd; a calibrated posterior has rms z near 1, and"
                     " holds the truth about 68% and 95% of the time)")

        import h5py
        with h5py.File(str(self.output), "w") as h5:
            h5["truth"], h5["mean"], h5["std"], h5["z"] = truth, mean, sd, z
            h5["inside68"], h5["inside95"] = inside[68], inside[95]
        channel.log()
        return 0


# end of file
