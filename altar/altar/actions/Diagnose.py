# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import numpy
import pathlib
# get the package
import altar
# the samples, as the forward check reads them
from .Forward import load_samples


# declaration
class Diagnose(altar.panel(), family='altar.actions.diagnose'):
    """
    Check how a run went: its annealing schedule, step by step, with the acceptance rate, the
    spread of the importance weights and the evidence, from the archived steps; and, given a
    reference posterior, e.g. from a longer run, how far the posterior is from it
    """


    # user configurable state
    theta = altar.properties.path(default="results/step_final.h5")
    theta.doc = "the final step of the run; the other archived steps are read from its directory"

    reference = altar.properties.path(default=None)
    reference.doc = "a reference posterior: an archived step, or a .txt/.h5 file with one " \
                    "sample per row"

    dataset = altar.properties.str(default=None)
    dataset.doc = "the dataset holding {reference} in a plain .h5 file; default the first one"

    output = altar.properties.path(default="diagnose.h5")
    output.doc = "the .h5 file for the schedule and the comparison"


    # commands
    @altar.export(tip="check the annealing schedule of a run, and its posterior against a reference")
    def default(self, plexus, **kwds):
        """
        Report the schedule of the run, and the distance of its posterior from the reference
        """
        import h5py
        model = plexus.model
        channel = plexus.info
        final = pathlib.Path(str(self.theta))
        schedule = self.schedule(final=final)
        channel.line(f"diagnose: '{final}', {len(schedule['beta'])} archived steps")
        channel.line(f"  {'step':>5s}{'beta':>12s}{'acceptance':>12s}{'scaling':>10s}"
                     f"{'cov(w)':>9s}{'ESS/N':>8s}{'log evidence':>16s}")
        for i, step in enumerate(schedule["step"]):
            channel.line(f"  {step:>5s}{schedule['beta'][i]:12.4g}{schedule['acceptance'][i]:12.3f}"
                         f"{schedule['scaling'][i]:10.4g}{schedule['cov'][i]:9.3f}"
                         f"{schedule['ess'][i]:8.3f}{schedule['evidence'][i]:16.6g}")
        channel.line("  (acceptance: of the moves in the step; cov(w), ESS/N: of the importance"
                     " weights that led to it)")

        with h5py.File(str(self.output), "w") as h5:
            for key in ["beta", "acceptance", "scaling", "cov", "ess", "evidence"]:
                h5[f"schedule/{key}"] = numpy.asarray(schedule[key], dtype=float)
            if self.reference is not None:
                self.compare(model=model, final=final, h5=h5, channel=channel)
        channel.log()
        return 0


    def schedule(self, final):
        """
        The archived steps of the run of {final}, with their statistics
        """
        import h5py
        files = sorted(final.parent.glob("step_[0-9]*.h5"))
        # the final step, unless it repeats the last numbered one
        if not files or self.beta(files[-1]) != self.beta(final):
            files.append(final)
        schedule = {key: [] for key in ["step", "beta", "acceptance", "scaling", "cov", "ess",
                                        "evidence"]}
        # the moves of each iteration, (iteration, beta, scaling, accepted, invalid, rejected)
        with h5py.File(final, "r") as h5:
            moves = numpy.asarray(h5["Statistics/mc_updates"]) if "Statistics/mc_updates" in h5 else None
        for path in files:
            with h5py.File(path, "r") as h5:
                β = float(numpy.asarray(h5["Annealer/beta"]))
                w = numpy.asarray(h5["Annealer/weights"]) if "Annealer/weights" in h5 else None
                evidence = (float(numpy.asarray(h5["Annealer/log_evidence"]))
                            if "Annealer/log_evidence" in h5 else numpy.nan)
            label = path.stem.split("_", 1)[1]
            row = None
            if moves is not None:
                # the steps are numbered by iteration; the final one is the last
                iteration = int(label) if label != "final" else moves.shape[0] - 1
                row = moves[iteration] if iteration < moves.shape[0] else None
            total = row[3:6].sum() if row is not None else 0
            schedule["step"].append(label)
            schedule["beta"].append(β)
            schedule["acceptance"].append(row[3] / total if total else numpy.nan)
            schedule["scaling"].append(row[2] if row is not None else numpy.nan)
            if w is not None and w.sum() > 0:
                w = w / w.mean()
                schedule["cov"].append(w.std())
                schedule["ess"].append(1 / (w ** 2).mean())
            else:
                schedule["cov"].append(numpy.nan)
                schedule["ess"].append(numpy.nan)
            schedule["evidence"].append(evidence)
        return schedule


    @staticmethod
    def beta(path):
        """
        The temperature of the archived step at {path}
        """
        import h5py
        with h5py.File(path, "r") as h5:
            return float(numpy.asarray(h5["Annealer/beta"]))


    def compare(self, model, final, h5, channel):
        """
        Compare the posterior of {final} with my reference, parameter set by parameter set
        """
        θ = load_samples(model=model, path=final)
        ref = load_samples(model=model, path=self.reference, dataset=self.dataset)
        sd = ref.std(axis=0)
        shift = numpy.abs(θ.mean(axis=0) - ref.mean(axis=0)) / numpy.where(sd > 0, sd, numpy.inf)
        ratio = θ.std(axis=0) / numpy.where(sd > 0, sd, numpy.nan)
        channel.line(f"  against '{self.reference}' ({ref.shape[0]} samples):")
        channel.line(f"  {'parameter set':16s}{'count':>6s}{'|dmean|/sd median':>19s}{'max':>7s}"
                     f"{'sd ratio median':>17s}{'[min, max]':>16s}")
        for name in model.psets_list:
            pset = model.psets[name]
            c = slice(pset.offset, pset.offset + pset.count)
            channel.line(f"  {name:16s}{pset.count:6d}{numpy.median(shift[c]):19.3f}"
                         f"{shift[c].max():7.3f}{numpy.nanmedian(ratio[c]):17.3f}"
                         f"{f'[{numpy.nanmin(ratio[c]):.3f}, {numpy.nanmax(ratio[c]):.3f}]':>16s}")
        channel.line("  (|dmean|/sd: the shift of the mean, in reference sd; sd ratio: this run's"
                     " sd over the reference's; below 1, too narrow)")
        h5["reference/shift"], h5["reference/sd_ratio"] = shift, ratio
        return


# end of file
