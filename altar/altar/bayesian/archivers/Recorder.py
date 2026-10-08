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
from __future__ import annotations
import typing
import numpy
# the package
import altar

if typing.TYPE_CHECKING:
    import h5py
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.shells.Application import Application


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
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me given an {application} context
        """

        # create a statistics list
        self.statistics = []
        # in-memory storage for recorded datasets
        self.records = {}
        # registered components to query at each save point
        self._components = []

        # all done
        return self


    @altar.export
    def record(self, step: BayesianState, iteration: int | None, psets: dict,
               **kwds) -> typing.Self:
        """
        Record the final state of the calculation
        """
        # delegate to the finalization hook
        self.final(step=step, iteration=None, psets=psets)
        # all done
        return self

    def recordstep(self, step: BayesianState, stats: dict, psets: dict) -> typing.Self:
        """
        Record step to file for ce
        """
        # record statistics information
        statcopy = stats.copy()
        self.statistics.append(statcopy)
        return self

    def save_stats(self) -> typing.Self:
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

    def start(self, step: BayesianState, iteration: int | None, psets: dict,
              **kwds) -> typing.Self:
        """
        Hook for the beginning of annealing.
        """
        self._set_context(iteration=iteration, psets=psets)
        return self

    def top(self, step: BayesianState, iteration: int | None, psets: dict,
            **kwds) -> typing.Self:
        """
        Hook for the top of a beta step.
        """
        self._set_context(iteration=iteration, psets=psets)
        return self

    def bottom(self, step: BayesianState, iteration: int | None, psets: dict,
               **kwds) -> typing.Self:
        """
        Hook for the bottom of a beta step.
        """
        if iteration % self.output_freq == 0:
            self._set_context(iteration=iteration, psets=psets)
            step.record(archiver=self)
            for component in self._components:
                component.record(archiver=self)
        return self

    def final(self, step: BayesianState, iteration: int | None, psets: dict,
              **kwds) -> typing.Self:
        """
        Hook for the end of annealing.
        """
        self.save_stats()
        self._set_context(iteration=None, psets=psets)
        step.record(archiver=self)
        for component in self._components:
            component.record(archiver=self)
        self._record_statistics()
        return self

    @altar.export
    def write(self, path: str, data: typing.Any, info: dict | None = None) -> typing.Self:
        """
        Store one dataset in memory.

        {path} is "Group/Name" or "Group/Sub/Name".  {data} is anything numpy can view as an
        array, or a scalar.  {info} is optional metadata stored alongside.
        """
        arr = numpy.asarray(data)
        key = self._current_label
        if key not in self.records:
            self.records[key] = []
        self.records[key].append((path, arr, info))
        return self

    @altar.export
    def register(self, component: typing.Any) -> typing.Self:
        """
        Register a component whose record(archiver) will be called at each save point.
        """
        self._components.append(component)
        return self

    def save(self, record: tuple[str, str, typing.Any]) -> typing.Self:
        """
        Compatibility shim: accept the old (group, name, data) tuple form.
        """
        group_name, dataset_name, data = record
        path = '/'.join(p for p in [group_name, dataset_name] if p)
        return self.write(path, data)

    def _record_statistics(self) -> typing.Self:
        """
        Persist the MC statistics table in memory.
        """
        stats = self._statistics_as_array()
        if stats is None:
            return self
        self.save(("Statistics", "mc_updates", stats))
        return self

    def _statistics_as_array(self) -> numpy.ndarray | None:
        """
        Pack statistics into a numeric array.
        """
        if not self.statistics:
            return None

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

    def _set_context(self, iteration: int | None, psets: dict) -> typing.Self:
        """
        Save context needed by components during recording.
        """
        self._current_label = 'final' if iteration is None else iteration
        self.psets = psets
        return self

    # unconfigurable traits
    statistics: list[dict] | None = None
    records: dict | None = None
    psets: dict | None = None
    _current_label: int | str | None = None
    _components: list = []

# end of file
