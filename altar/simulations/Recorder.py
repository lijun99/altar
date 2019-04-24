# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#
# Author(s): michael a.g. aïvázis, Lijun Zhu


# the package
import altar
# my protocol
from .Archiver import Archiver as archiver


# an implementation of the archiver protocol
class Recorder(altar.component, family="altar.simulations.archivers.recorder", implements=archiver):
    """
    Recorder stores the intermediate simulation state in memory
    """

    # user configurable state
    theta = altar.properties.path(default="theta.txt")
    theta.doc = "the path to the file with the final posterior sample"

    sigma = altar.properties.path(default="sigma.txt")
    sigma.doc = "the path to the file with the final parameter correlation matrix"

    llk = altar.properties.path(default="llk.txt")
    llk.doc = "the path to the file with the final posterior log likelihood"

    output_dir = altar.properties.path(default="results")
    output_dir.doc = "the directory to save results"

    output_freq = altar.properties.int(default=1)
    output_freq.doc = "the frequency to write step data to files"

    statistics = None

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
    def record(self, step, iteration, **kwds):
        """
        Record the final state of the calculation
        """
        # record the samples
        #step.theta.save(filename=self.theta)
        # the covariance matrix
        #step.sigma.save(filename=self.sigma)
        # and the posterior log likelihood
        #step.posterior.save(filename=self.llk)

        # save the statistics
        self.saveStats()
        # save the last step
        step.save_hdf5(path=self.output_dir, iteration=None)
        # all done
        return self

    def recordstep(self, step, stats):
        """
        Record step to file for ce 
        """
        iteration = stats['iteration']
        # output CoolingStep 
        if iteration%self.output_freq == 0:
            step.save_hdf5(path=self.output_dir, iteration=iteration)
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
        statfile.write("iteration, beta, scaling, (accepted, rejected, invalid)\n")
        # write the statistics
        for item in self.statistics:
            stats = [str(k) for k in item.values()]
            statfile.writelines(", ".join(stats) + "\n")
        # close the file
        statfile.close()
        #return
        return self


# end of file
