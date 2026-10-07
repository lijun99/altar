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
import itertools
import numpy
# the package
import altar
# my protocol
from .Scheduler import Scheduler as scheduler
from ..statistics import weighted_covariance


# declaration
class COV(altar.component, family="altar.schedulers.cov", implements=scheduler):
    r"""
    Annealing schedule based on attaining a particular value for the coefficient of variation
    (COV) of the data likelihood; after Ching[2007].

    The goal is to compute a proposed update δβ_m to the temperature β_m such that the vector
    of weights w_m given by

        w_m := π(D|θ_m)^{δβ_m}

    has a particular target value for

        COV(w_m) := \sqrt{<(w_m-<w_m>)^2>} / <w_m>
    """

    # user configurable state
    target = altar.properties.float(default=1.0)
    target.doc = 'the target value for COV'

    solver = altar.bayesian.solver()
    solver.doc = 'the δβ solver'

    check_positive_definiteness = altar.properties.bool(default=True)
    check_positive_definiteness.doc = 'whether to check the positive definiteness of Σ matrix and condition it accordingly'

    min_eigenvalue_ratio = altar.properties.float(default=0.001)
    min_eigenvalue_ratio.doc = 'the desired minimal eigenvalue of Σ matrix, as a ratio to the max eigenvalue'

    beta_resampling_start = altar.properties.float(default=0)
    beta_resampling_start.doc = 'the beta threshold to start the resampling procedure'

    beta_min = altar.properties.float(default=0)
    beta_min.doc = 'the minimum beta value to be used'

    use_low_variance_resampler = altar.properties.bool(default=False)
    use_low_variance_resampler.doc = "whether to equal spaced random numbers for resampling"

    # public data
    cov = 0.0 # the actual value for COV we were able to attain


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        # get the rng wrapper
        self.rng = application.rng.rng

        # initialize my solver
        self.solver.initialize(application=application, scheduler=self)

        # set up the distribution for building the sample multiplicities
        self.uniform = altar.pdf.uniform(support=(0,1), rng=self.rng)

        # grab the info channel
        self.info = application.info

        # all done
        return self


    @altar.export
    def update(self, step):
        """
        Push {step} forward along the annealing schedule
        """

        # get the new temperature and store it
        β = self.update_temperature(step=step)
        # the samples the weights belong to, for the proposal covariance: those before resampling
        step.weighted_theta = None
        # resampling according to their likelihood
        if β > self.beta_resampling_start:
            step.weighted_theta = getattr(step, "theta_sampling", step.theta).clone()
            θ, (prior, data, posterior), θ_sampling, jacobian = self.resampling(step=step)
            # update the step after the resampling
            step.prior.copy(prior)
            step.data.copy(data)
            step.theta.copy(θ)
            # a reparameterized step keeps theta_sampling/jacobian as separate buffers (see
            # {altar.bayesian.states.CoolingStep}), not aliases of theta -- they must be
            # reordered by the exact same sample indices, or theta/theta_sampling end up
            # describing different chains after resampling (row i's physical theta no longer
            # corresponds to row i's sampling-space theta), silently corrupting any
            # reparameterized gradient-based sampler (e.g. HMC) that reads both
            if getattr(step, 'has_reparametrization', False):
                step.theta_sampling.copy(θ_sampling)
                if jacobian is not None and step.jacobian is not None:
                    step.jacobian.copy(jacobian)

        # update the step (common procedures with or w/o resampling)
        step.beta = β

        # recompute posterior with updated beta
        step.compute_posterior()

        # and return it
        return step


    @altar.export
    def update_temperature(self, step):
        """
        Generate the next temperature increment
        """
        # grab the data log-likelihood
        data_likelihood = step.data
        # initialize the vector of weights
        w = altar.vector(shape=step.samples).zero()
        # compute {δβ} and the normalized {w}
        β, self.cov = self.solver.solve(data_likelihood, w)
        # publish weights on the step so all components (proposal, etc.) can consume them
        step.weights = w
        # adjust β if it is too small
        β = max(β, self.beta_min)
        # and return the new temperature
        return β


    @altar.export
    def compute_covariance(self, step):
        r"""
        Compute the parameter covariance Σ of the sample in {step}

          Σ = c_m^2 \sum_{i \in samples} \tilde{w}_{i} θ_i θ_i^T} - \bar{θ} \bar{θ}^Τ

        where

          \bar{θ} = \sum_{i \in samples} \tilde{w}_{i} θ_{i}

        The covariance Σ gets used to build a proposal pdf for the posterior
        """
        # unpack what i need
        w = step.weights # published by update_temperature(); assumed normalized
        θ = step.theta # the current sample set
        # extract the number of samples and number of parameters
        samples = step.samples
        parameters = step.parameters

        # check the geometries
        assert w.shape == samples
        assert θ.shape == (samples, parameters)

        # the weighted outer products about the weighted mean, in one matrix product
        Σ = altar.matrix(shape=(parameters, parameters))
        Σ.ndarray()[:] = weighted_covariance(θ.ndarray(), w.ndarray())

        # condition the covariance matrix
        if self.check_positive_definiteness:
            self.condition_covariance(Σ=Σ)

        # all done
        return Σ


    @altar.export
    def rank(self, step):
        """
        Rebuild the sample and its statistics sorted by the likelihood of the parameter values
        """
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
        multi = self.compute_sample_multiplicities(step=step).counts()
        # print("      histogram as vector:")
        # print("        counts: {}".format(tuple(multi)))

        # compute the permutation that would sort the frequency table according to the sample
        # multiplicity, in reverse order
        p = multi.sortIndirect().reverse()
        # print("        sorted: {}".format(tuple(p[i] for i in range(p.shape))))

        # the number of samples we have processed
        done = 0
        # start moving stuff around until we have built a complete sample set
        for i in range(p.shape):
            # the old sample index
            old = p[i]
            # and its multiplicity
            count = int(multi[old])
            # if the count has dropped to zero, we are done
            if count == 0: break
            # otherwise, duplicate this sample {count} times
            for dupl in range(count):
                # update the samples
                for param in range(step.parameters):
                    θ[done, param] = θOld[old, param]
                # update the log-likelihoods
                prior[done] = priorOld[old]
                data[done] = dataOld[old]
                posterior[done] = postOld[old]
                # update the number of processed samples
                done += 1
                # print(i, old, count, done)

        # return the shuffled data
        return θ, (prior, data, posterior)

    # important resampling: rank is not needed, shuffle is recommended
    def resampling(self, step):
        """
        Rebuild the sample and its statistics sorted by the likelihood of the parameter values
        """
        θOld = step.theta
        priorOld = step.prior
        dataOld = step.data
        postOld = step.posterior
        # allocate the new entities
        θ = altar.matrix(shape=θOld.shape)
        prior = altar.vector(shape=priorOld.shape)
        data = altar.vector(shape=dataOld.shape)
        posterior = altar.vector(shape=postOld.shape)

        # a reparameterized step carries theta_sampling/jacobian as separate buffers from
        # theta (not aliases; see {altar.bayesian.states.CoolingStep}), so they must be
        # reordered by the exact same sample indices below, or theta/theta_sampling end up
        # describing different chains after resampling
        has_reparametrization = getattr(step, 'has_reparametrization', False)
        θSamplingOld = θSampling = jacobianOld = jacobian = None
        if has_reparametrization:
            θSamplingOld = step.theta_sampling
            θSampling = altar.matrix(shape=θSamplingOld.shape)
            jacobianOld = step.jacobian
            if jacobianOld is not None:
                jacobian = altar.vector(shape=jacobianOld.shape)

        # build a histogram for the new samples and convert it into a vector
        multi = self.compute_sample_multiplicities(step=step).counts()
        # print("      histogram as vector:")
        # print("        counts: {}".format(tuple(multi)))

        counts = multi.ndarray().astype(int)
        unique_samples = int(numpy.count_nonzero(counts))
        # the index of the old sample behind each new one, duplicated by its count
        indices = altar.vector(shape=multi.shape)
        indices.ndarray()[:] = numpy.repeat(numpy.arange(counts.size), counts)
        # shuffle the indices
        indices.shuffle(rng=self.rng)
        rows = indices.ndarray().astype(int)

        self.info.log(f"resampling: unique samples {unique_samples} out of {multi.shape}")

        # copy theta, (prior, data, posterior) over according to the indices, and
        # theta_sampling/jacobian, when reparameterized
        pairs = [(θ, θOld), (prior, priorOld), (data, dataOld), (posterior, postOld)]
        if has_reparametrization:
            pairs.append((θSampling, θSamplingOld))
            if jacobian is not None:
                pairs.append((jacobian, jacobianOld))
        for new, old in pairs:
            new.ndarray()[:] = old.ndarray()[rows]

        # return the shuffled data
        return θ, (prior, data, posterior), θSampling, jacobian


    # implementation details
    def condition_covariance(self, Σ):
        """
        Make sure the covariance matrix Σ is symmetric and positive definite
        """
        # replaces negative or small eigenvalues with min_eigenvalue_ratio*max_eigenvalue
        altar.libaltar.matrix_condition(Σ, self.min_eigenvalue_ratio)
        # all done
        return Σ


    def compute_sample_multiplicities(self, step):
        """
        Prepare a frequency vector for the new samples given the scaled data log-likelihood in
        {w} for this cooling step
        """
        # print("    computing sample multiplicities:")
        # unpack what we need
        w = step.weights
        samples = step.samples

        # build a vector of random numbers uniformly distributed in [0,1]
        r = altar.vector(shape=samples)
        if self.use_low_variance_resampler:
            # use equal spaced random number s+i/samples in [0, 1]
            altar.libaltar.low_variance_random(self.rng, r)
        else:
            # use uniform pdf generator in [0, 1]
            r.random(pdf=self.uniform)

        # compute the bin edges in the range [0, 1]
        ticks = tuple(self.build_histogram_ranges(w))
        # build a histogram
        h = altar.histogram(bins=samples).ranges(edges=ticks).fill(r)
        # and return it
        return h


    def build_histogram_ranges(self, w):
        """
        Build histogram bins based on the scaled data log-likelihood
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
