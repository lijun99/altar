# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import os
import re
import numpy


# declaration
class Checkpoint:
    """
    The state of an annealing run as one of its archived steps records it, e.g. a
    {step_015.h5} of the h5 recorder: the temperature, the physical samples and their data
    likelihoods, and the history of the run up to that step
    """


    # interface
    def rows(self, total, start, count):
        """
        The physical samples and data likelihoods of rows {start} to {start + count} of a
        population of {total}; a file with a different population size is drawn from, with
        replacement, the same way on every task
        """
        # the rows of the file that make up the population
        if total == self.samples:
            # all of them, in order
            index = numpy.arange(total)
        else:
            # a population at the end of a step is equally weighted, so any draw represents it
            index = numpy.random.default_rng(seed=0).integers(0, self.samples, size=total)
        # my share of them
        mine = index[start:start + count]
        # all done
        return self.theta[mine], self.data[mine]


    def history(self):
        """
        The statistics of the steps up to mine, the way an archiver keeps them
        """
        # one entry per row of the table
        return [
            {"iteration": int(row[0]), "beta": float(row[1]), "scaling": float(row[2]),
             "stats": (int(row[3]), int(row[4]), int(row[5]))}
            for row in self.statistics
        ]


    # meta-methods
    def __init__(self, path, model, **kwds):
        # chain up
        super().__init__(**kwds)
        # support
        import h5py
        # the file
        self.path = str(path)
        with h5py.File(self.path, "r") as f:
            # the temperature
            self.beta = float(numpy.asarray(f["Annealer/beta"]))
            # the physical samples, a parameter set at a time, where the model puts them
            self.theta = self._samples(group=f["ParameterSets"], model=model)
            self.samples = self.theta.shape[0]
            # their data likelihoods
            self.data = numpy.asarray(f["Bayesian/likelihood"], dtype="float64")
            # the statistics of the steps so far: iteration, beta, scaling, and the counts
            self.statistics = numpy.asarray(f["Statistics/mc_updates"]) \
                if "Statistics/mc_updates" in f else numpy.zeros((0, 6))
        # the iteration this step concluded, from the statistics, else from the file name
        if len(self.statistics):
            self.iteration = int(self.statistics[-1, 0])
        else:
            match = re.fullmatch(r"step_(\d+)\.h5", os.path.basename(self.path))
            self.iteration = int(match.group(1)) if match else 0
            # older step files left the statistics to the BetaStatistics.txt of a finished run
            self.statistics = self._table(iteration=self.iteration)
        # and the proposal scaling it left behind, if it recorded one
        self.scaling = float(self.statistics[-1, 2]) if len(self.statistics) else None
        # all done
        return


    # implementation details
    def _samples(self, group, model):
        """
        Assemble the physical samples from the parameter sets of the {group}
        """
        # without parameter sets, the file holds the sample matrix itself
        if "theta" in group or "theta_sampling" in group:
            name = "theta" if "theta" in group else "theta_sampling"
            theta = numpy.asarray(group[name], dtype="float64")
        else:
            theta = None
            for name in model.psets_list:
                pset = model.psets[name]
                # the physical values, which a run without reparameterization keeps as sampled
                key = f"{name}_physical" if f"{name}_physical" in group else f"{name}_sampling"
                if key not in group:
                    raise ValueError(f"{self.path}: no samples of the parameter set '{name}'")
                values = numpy.asarray(group[key], dtype="float64")
                if values.shape[1] != pset.count:
                    raise ValueError(f"{self.path}: '{name}' has {values.shape[1]} parameters, "
                                     f"the model expects {pset.count}")
                if theta is None:
                    theta = numpy.zeros((values.shape[0], model.parameters))
                theta[:, pset.offset:pset.offset + pset.count] = values
        # the samples must fit the model
        if theta.shape[1] != model.parameters:
            raise ValueError(f"{self.path}: {theta.shape[1]} parameters, the model expects "
                             f"{model.parameters}")
        # all done
        return theta


    def _table(self, iteration):
        """
        The rows up to {iteration} of the BetaStatistics.txt next to my file, if there is one
        """
        table = os.path.join(os.path.dirname(self.path), "BetaStatistics.txt")
        if not os.path.exists(table):
            return numpy.zeros((0, 6))
        rows = numpy.atleast_2d(numpy.loadtxt(table, delimiter=",", skiprows=1)).reshape(-1, 6)
        return rows[rows[:, 0] <= iteration]


# end of file
