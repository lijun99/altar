#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# lijun zhu <ljzhu@gps.caltech.edu>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# externals
import itertools
# the package
import altar

# declaration
class ImportanceResampler(altar.component, family="altar.bayesian.importanceresampler"):
    """
    Implementation of importance resampling strategy for Bayesian inference.
    """

    # user configurable state
    use_low_variance_resampler = altar.properties.bool(default=False)
    use_low_variance_resampler.doc = "whether to use equal spaced random numbers for resampling"

    beta_resampling_start = altar.properties.float(default=0)
    beta_resampling_start.doc = 'the beta threshold to start the resampling procedure'

    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        # get the rng wrapper
        self.rng = application.rng.rng

        # set up the distribution for building the sample multiplicities
        self.uniform = altar.pdf.uniform(support=(0,1), rng=self.rng)

        # grab the info channel
        self.info = application.info

        # all done
        return self

    @altar.export
    def resample(self, w, step, β):
        """
        Rebuild the sample and its statistics based on importance weights if β threshold is met
        """
        # check if resampling should be performed
        if β <= self.beta_resampling_start:
            return False
            
        θOld = step.theta
        priorOld = step.prior
        dataOld = step.data
        postOld = step.posterior
        # allocate the new entities
        θ = altar.matrix(shape=θOld.shape)
        prior = altar.vector(shape=priorOld.shape)
        data = altar.vector(shape=dataOld.shape)
        posterior = altar.vector(shape=postOld.shape)

        # build a histogram for the new samples and convert it into a vector
        multi = self.computeSampleMultiplicities(w=w, step=step).values()

        # unique samples count
        unique_samples = 0
        # sample count
        index = 0
        # indices for kept samples
        indices = altar.vector(shape=multi.shape)

        # record kept sample indices
        for i in range(multi.shape):
            count = int(multi[i])
            # if count is zero, skip
            if count == 0: continue
            # add the unique samples count
            unique_samples += 1
            # duplicate indices
            for ic in range(count):
                indices[index] = i
                index += 1
        # shuffle the indices
        indices.shuffle(rng=self.rng)

        self.info.log(f"resampling: unique samples {unique_samples} out of {multi.shape}")

        # copy theta, (prior, data, posterior) over according to the indices
        for i in range(indices.shape):
            # get the index for old samples
            old = int(indices[i])
            # duplicate theta
            for param in range(step.parameters):
                θ[i, param] = θOld[old, param]
            prior[i] = priorOld[old]
            data[i] = dataOld[old]
            posterior[i] = postOld[old]

        # update the step with resampled data
        step.prior.copy(prior)
        step.data.copy(data)
        step.theta.copy(θ)

        # indicate resampling was performed
        return True

    def computeSampleMultiplicities(self, w, step):
        """
        Prepare a frequency vector for the new samples given the importance weights {w}
        """
        # unpack what we need
        samples = step.samples

        # build a vector of random numbers uniformly distributed in [0,1]
        r = altar.vector(shape=samples)
        if self.use_low_variance_resampler:
            # use equal spaced random number s+i/samples in [0, 1]
            altar.libaltar.low_variance_random(self.rng.rng, r.data)
        else:
            # use uniform pdf generator in [0, 1]
            r.random(pdf=self.uniform)

        # compute the bin edges in the range [0, 1]
        ticks = tuple(self.buildHistogramRanges(w))
        # build a histogram
        h = altar.histogram(bins=samples).ranges(points=ticks).fill(r)
        # and return it
        return h

    def buildHistogramRanges(self, w):
        """
        Build histogram bins based on the importance weights
        """
        # start at 0
        yield 0
        # yield the partial sums
        for partialSum in itertools.accumulate(w): yield partialSum
        # all done
        return

    # private data
    uniform = None
    rng = None

# end of file
