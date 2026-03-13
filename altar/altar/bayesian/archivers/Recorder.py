# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): michael a.g. aïvázis, Lijun Zhu


# the package
import altar


# an implementation of the archiver protocol
class Recorder(
                altar.component,
                family="altar.simulations.archivers.recorder",
                implements=altar.simulations.archiver):
    """
    Recorder stores the intermediate simulation state in memory
    """

    # user configurable traits
    output_dir = altar.properties.path(default="results")
    output_dir.doc = "the directory to save results"

    output_freq = altar.properties.int(default=1)
    output_freq.doc = "the frequency to write step data to files"

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize me given an {application} context
        """

        # create a statistics list
        self.statistics= []
        # in-memory storage for recorded datasets
        self.records = {}

        # all done
        return self


    @altar.export
    def record(self, step, iteration, psets, **kwds):
        """
        Record the final state of the calculation
        """
        # delegate to the finalization hook
        self.final(step=step, iteration=None, psets=psets)
        # all done
        return self

    def recordstep(self, step, stats, psets):
        """
        Record step to file for ce
        """
        # record statistics information
        statcopy = stats.copy()
        self.statistics.append(statcopy)
        return self

    def save_stats(self):
        """
        Save the statistics information to file
        """
        # output filename
        import os
        filename = os.path.join(self.output_dir.path, "BetaStatistics.txt")
        # ensure destination directory exists
        os.makedirs(self.output_dir.path, exist_ok=True)
        # open the file
        statfile = open(filename, "w")
        # write the header
        statfile.write("iteration, beta, scaling, (accepted, invalid, rejected)\n")
        # write the statistics
        for item in self.statistics:
            stats = item.get("stats") or (0, 0, 0)
            line = [
                item.get("iteration"),
                item.get("beta"),
                item.get("scaling"),
                stats[0],
                stats[1],
                stats[2],
            ]
            statfile.writelines(", ".join(str(value) for value in line) + "\n")
        # close the file
        statfile.close()
        #return
        return self

    def start(self, step, iteration, psets, **kwds):
        """
        Hook for the beginning of annealing.
        """
        self._set_context(iteration=iteration, psets=psets)
        return self

    def top(self, step, iteration, psets, **kwds):
        """
        Hook for the top of a beta step.
        """
        self._set_context(iteration=iteration, psets=psets)
        return self

    def bottom(self, step, iteration, psets, **kwds):
        """
        Hook for the bottom of a beta step.
        """
        if iteration % self.output_freq == 0:
            self._set_context(iteration=iteration, psets=psets)
            step.record(archiver=self)
        return self

    def final(self, step, iteration, psets, **kwds):
        """
        Hook for the end of annealing.
        """
        self.save_stats()
        self._set_context(iteration=None, psets=psets)
        step.record(archiver=self)
        self._record_statistics()
        return self

    def save(self, record):
        """
        Store a dataset tuple in memory.
        """
        key = self._current_label
        if key not in self.records:
            self.records[key] = []
        self.records[key].append(record)
        return self

    def _record_statistics(self):
        """
        Persist the MC statistics table in memory.
        """
        stats = self._statistics_as_array()
        if stats is None:
            return self
        self.save(("Statistics", "mc_updates", stats))
        return self

    def _statistics_as_array(self):
        """
        Pack statistics into a numeric array.
        """
        if not self.statistics:
            return None
        import numpy

        rows = []
        for item in self.statistics:
            stats = item.get("stats") or (0, 0, 0)
            accepted = stats[0]
            invalid = stats[1]
            rejected = stats[2]
            rows.append([
                item.get("iteration"),
                item.get("beta"),
                item.get("scaling"),
                accepted,
                invalid,
                rejected,
            ])
        return numpy.asarray(rows)

    def _set_context(self, iteration, psets):
        """
        Save context needed by components during recording.
        """
        self._current_label = 'final' if iteration is None else iteration
        self.psets = psets
        return self

    # unconfigurable traits
    statistics = None
    records = None
    psets = None
    _current_label = None

# end of file
