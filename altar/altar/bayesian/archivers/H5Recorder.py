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
class H5Recorder(
                altar.component,
                family="altar.simulations.archivers.h5recorder",
                implements=altar.simulations.archiver):
    """
    H5Recorder stores the intermediate simulation state to HDF5 files
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
        # clear any open file handles
        self._file = None
        self._groups = None

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
        filename = "BetaStatistics.txt"
        # ensure destination directory exists
        output_dir = self.output_dir.path if isinstance(self.output_dir, altar.primitives.path) else self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)
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
            self._open_file(iteration=iteration)
            step.record(archiver=self)
            self._close_file()
        return self

    def final(self, step, iteration, psets, **kwds):
        """
        Hook for the end of annealing.
        """
        self.save_stats()
        self._set_context(iteration=None, psets=psets)
        self._open_file(iteration=None)
        step.record(archiver=self)
        self._record_statistics()
        self._close_file()
        return self

    def save(self, record):
        """
        Persist a dataset tuple into the current HDF5 file.
        """
        if self._file is None:
            raise RuntimeError("H5Recorder.save called without an open file")
        group_name, dataset_name, data = record
        if group_name in ("", None, "/"):
            group = self._file
        else:
            group = self._file.require_group(group_name)
        parts = [part for part in dataset_name.split("/") if part]
        if not parts:
            raise ValueError("empty dataset name")
        for part in parts[:-1]:
            group = group.require_group(part)
        group.create_dataset(parts[-1], data=data)
        return self

    def _record_statistics(self):
        """
        Persist the MC statistics table into the current HDF5 file.
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

    def _open_file(self, iteration):
        """
        Open a fresh HDF5 file for the current record.
        """
        import os
        import h5py

        str_iteration = 'final' if iteration is None else str(iteration).zfill(3)
        str_path = self.output_dir.path if isinstance(self.output_dir, altar.primitives.path) else self.output_dir
        if not os.path.exists(str_path):
            os.makedirs(str_path)
        filename = os.path.join(str_path, "step_"+str_iteration+".h5")
        self._file = h5py.File(filename, 'w')
        self._groups = {}
        return self

    def _close_file(self):
        """
        Close the current HDF5 file.
        """
        if self._file is not None:
            self._file.close()
        self._file = None
        self._groups = None
        return self

    def _set_context(self, iteration, psets):
        """
        Save context needed by components during recording.
        """
        self.psets = psets
        self._current_label = 'final' if iteration is None else iteration
        return self

    # unconfigurable traits
    statistics = None
    psets = None
    _current_label = None
    _file = None
    _groups = None

# end of file
