# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
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

        # all done
        return self


    @altar.export
    def record(self, step, iteration, psets, **kwds):
        """
        Record the final state of the calculation
        """
        # save the statistics
        self.saveStats()
        # save the last step
        step.save_hdf5(path=self.output_dir, iteration=None, psets=psets)
        # all done
        return self

    def recordstep(self, step, stats, psets):
        """
        Record step to file for ce
        """
        iteration = stats['iteration']

        # output CoolingStep
        if iteration%self.output_freq == 0:
            step.save_hdf5(path=self.output_dir, iteration=iteration, psets=psets)

        # record statistics information
        statcopy = stats.copy()
        self.statistics.append(statcopy)
        return self

    def saveStats(self):
        """
        Save the statistics information to file
        """
        # output filename
        filename = "BetaStatistics.txt"
        # open the file
        statfile = open(filename, "w")
        # write the header
        statfile.write("iteration, beta, scaling, (accepted, invalid, rejected)\n")
        # write the statistics
        for item in self.statistics:
            stats = [str(k) for k in item.values()]
            statfile.writelines(", ".join(stats) + "\n")
        # close the file
        statfile.close()
        #return
        return self

    # unconfigurable traits
    statistics = None

# end of file
