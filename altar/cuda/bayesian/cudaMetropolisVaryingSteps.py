# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# externals
import math
# the package
import altar
import altar.cuda
from altar.cuda import curand
from altar.cuda import cublas
from altar.cuda import libcudaaltar

# my protocol
from altar.bayesian.Sampler import Sampler


# declaration
class cudaMetropolisVaryingSteps(altar.component, family="altar.samplers.metropolis", implements=Sampler):
    """
    The Metropolis algorithm as a sampler of the posterior distribution
    """

    # types
    from .cudaCoolingStep import cudaCoolingStep


    # user configurable state
    scaling = altar.properties.float(default=.1)
    scaling.doc = 'the parameter covariance Σ is scaled by the square of this'

    acceptanceWeight = altar.properties.float(default=8)
    acceptanceWeight.doc = 'the weight of accepted samples during covariance rescaling'

    rejectionWeight = altar.properties.float(default=1)
    rejectionWeight.doc = 'the weight of rejected samples during covariance rescaling'

    max_mc_steps = altar.properties.int(default=100000)
    max_mc_steps.doc = 'the maximum Monte-Carlo steps for one beta step'

    corr_check_steps = altar.properties.int(default=1000)
    corr_check_steps.doc = 'the Monte-Carlo steps to compute the de'

    target_correlation = altar.properties.float(default=0.6)
    target_correlation.doc = 'the threshold of correlation to stop the chain'


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        # pull the chain length from the job specification
        self.mcsteps = application.job.steps

        # get the capsule of the random number generator

        self.device = application.controller.worker.device
        self.curng = self.device.curand_generator
        self.precision = application.job.gpuprecision

        # all done
        return self

    def cuInitialize(self, application):
        self.initialize(application=application)
        return self

    @altar.export
    def samplePosterior(self, annealer, step):
        """
        Sample the posterior distribution
        Arguments:
            annealer - the controller
            step - cpu CoolingStep
        Return:
            statistics (accepted/rejected/invalid) or (accepted/unlikely/rejected)
        """
        # grab the dispatcher
        dispatcher = annealer.dispatcher
        # notify we have started sampling the posterior
        dispatcher.notify(event=dispatcher.samplePosteriorStart, controller=annealer)

        # prepare the sampling pdf, copy step to gpu step
        self.prepareSamplingPDF(annealer=annealer, step=step)

        # check whether model parameters needed to be updated, e.g., Cp
        model = annealer.model

        if model.updateModel(annealer=annealer):
            # if updated, recompute datalikelihood and posterior
            gstep = self.gstep
            batch = gstep.samples
            gstep.prior.zero(), gstep.data.zero(), gstep.posterior.zero()
            model.likelihoods(annealer=annealer, step=gstep, batch=batch)

        # walk the chains
        statistics = self.walkChains(annealer=annealer, step=self.gstep)

        # finish the sampling pdf, copy gpu step back
        self.finishSamplingPDF(step=step)

        # notify we are done sampling the posterior
        dispatcher.notify(event=dispatcher.samplePosteriorFinish, controller=annealer)
        # all done
        return statistics


    @altar.provides
    def resample(self, annealer, statistics):
        """
        Update my statistics based on the results of walking my Markov chains
        """
        # update the scaling of the parameter covariance matrix
        self.adjustCovarianceScaling(*statistics)
        # all done
        return


    # implementation details
    def prepareSamplingPDF(self, annealer, step):
        """
        Re-scale and decompose the parameter covariance matrix, in preparation for the
        Metropolis update
        """
        # get the dispatcher
        dispatcher = annealer.dispatcher
        # notify we have started preparing the sampling PDF
        dispatcher.notify(event=dispatcher.prepareSamplingPDFStart, controller=annealer)

        # allocate local gpu data if not allocated
        self.gstep = annealer.worker.gstep
        if self.ginit is not True:
            self.allocateGPUData(step.samples, step.parameters)

        # copy cpu step state
        self.gstep.copyFromCPU(step=step)

        # unpack what i need
        self.gsigma_chol.copy_from_host(source=step.sigma)

        # compute its Cholesky decomposition
        self.gsigma_chol.Cholesky(uplo=cublas.FillModeUpper)

        # scale it
        self.gsigma_chol *= self.scaling

        # notify we are done preparing the sampling PDF
        dispatcher.notify(event=dispatcher.prepareSamplingPDFFinish, controller=annealer)
        # all done
        return

    def finishSamplingPDF(self, step):
        """
        procedures after sampling, e.g, copy data back to cpu
        """
        # copy gpu step back to cpu
        self.gstep.copyToCPU(step=step)
        return

    def walkChains(self, annealer, step):
        """
        Run the Metropolis algorithm on the Markov chains
        Arguments:
            annealer: cudaAnnealer
            step: cudaCoolingStep
        Return:
            statistics = (accepted, rejected, unlikely)
        """
        # get the model
        model = annealer.model
        # and the event dispatcher
        dispatcher = annealer.dispatcher

        # unpack what i need from the cooling step
        β = step.beta
        θ = step.theta
        prior = step.prior
        data = step.data
        posterior = step.posterior
        # get the parameter covariance
        Σ_chol = self.gsigma_chol
        # the sample geometry
        samples = step.samples
        parameters = step.parameters
        # a couple of functions from the math module

        # reset the accept/reject counters
        # note the difference from CPU Metropolis
        # invalid is for proposed samples out of range
        # accepted is for samples being updated
        # rejected is for samples rejected by M-H proposals
        accepted = rejected = invalid = 0

        # allocate some vectors that we use throughout the following
        # candidate likelihoods
        candidate = self.gcandidate
        θproposal = self.gproposal
        cprior = candidate.prior
        cdata = candidate.data
        cpost = candidate.posterior
        cθ = candidate.theta

        # the mask of samples rejected due to model constraint violations
        invalid_flags = self.ginvalid_flags
        valid_indices = self.gvalid_indices
        acceptance_flags = self.gacceptance_flags
        valid_samples = self.gvalid_samples
        # and a vector with random numbers for the Metropolis acceptance
        dice = self.gdice

        # copy the beta over
        candidate.beta = step.beta

        # make a copy of the starting samples
        θstart = θ.clone()
        correlation = 1.0
        mcsteps = 0

        while correlation > self.target_correlation and mcsteps < self.max_mc_steps:

            for ihop in range(self.corr_check_steps):
                # notify we are advancing the chains
                dispatcher.notify(event=dispatcher.chainAdvanceStart, controller=annealer)

                # notify we are starting the verification process
                dispatcher.notify(event=dispatcher.verifyStart, controller=annealer)

                # make a loop to make sure there is at least one new sample,
                # or certain numbers of new samples
                while True:
                    # the random displacement may have generated candidates that are outside the
                    # support of the model, so we must give it an opportunity to reject them;
                    # initialize the candidate sample by randomly displacing the current one
                    self.displace(displacement=θproposal)
                    θproposal += θ

                    # reset the mask and ask the model to verify the sample validity
                    # note that I have redefined model.verify to use theta as input

                    model.cuVerify(theta=θproposal, mask=invalid_flags.zero())

                    invalid_step = int(invalid_flags.sum())
                    valid = samples - invalid_step
                    # if valid > 0, continue; otherwise go back to repropose new samples
                    if valid > 1 :
                        break

                # set indices for valid samples, return valid samples count
                libcudaaltar.cudaMetropolis_setValidSampleIndices(valid_indices.data, invalid_flags.data,
                                                                  valid_samples.data)

                # get the invalid samples count
                invalid += invalid_step

                # queue valid samples to first rows of cθ
                libcudaaltar.cudaMetropolis_queueValidSamples(cθ.data, θproposal.data, valid_indices.data, valid)

                # notify that the verification process is finished
                dispatcher.notify(event=dispatcher.verifyFinish, controller=annealer)

                # initialize the likelihoods
                likelihoods = cprior.zero(), cdata.zero(), cpost.zero()

                # compute the probabilities/likelihoods
                model.likelihoods(annealer=annealer, step=candidate, batch=valid)

                # randomize the Metropolis acceptance vector
                dice = curand.uniform(self.curng, out=dice)

                # notify we are starting accepting samples
                dispatcher.notify(event=dispatcher.acceptStart, controller=annealer)

                # accept/reject: go through all the samples
                libcudaaltar.cudaMetropolis_metropolisUpdate(
                    θ.data, prior.data, data.data, posterior.data, # original
                    cθ.data, cprior.data, cdata.data, cpost.data,  # candidate
                    dice.data, acceptance_flags.zero().data, valid_indices.data, valid)

                # counting the acceptance/rejection
                accepted_step = int(acceptance_flags.sum())
                accepted += accepted_step
                rejected += valid - accepted_step

            # notify we are done accepting samples
            dispatcher.notify(event=dispatcher.acceptFinish, controller=annealer)

            # notify we are done advancing the chains
            dispatcher.notify(event=dispatcher.chainAdvanceFinish, controller=annealer)

            correlation = altar.cuda.stats.correlation(θstart, θ, axis=0).amax()
            mcsteps += self.corr_check_steps
            print(f"correlation {correlation} at {mcsteps}")

        # all done
        # print(f'stats: accepted {accepted}, invalid {invalid}, rejected {rejected}')
        return accepted, invalid, rejected


    def displace(self, displacement):
        """
        Construct a set of displacement vectors for the random walk from a distribution with zero
        mean and my covariance
        """
        # get my decomposed covariance
        Σ_chol = self.gsigma_chol

        # generate gaussian random numbers (samples x parameters)
        altar.cuda.curand.gaussian(out=displacement)

        # multiply by the sigma_cholesky
        cublas.trmm(Σ_chol, displacement, out=displacement,
            alpha=1.0, uplo=cublas.FillModeUpper, side=cublas.SideRight,
            transa = cublas.OpNoTrans, diag=cublas.DiagNonUnit)
        # and return
        return displacement


    def adjustCovarianceScaling(self, accepted, invalid, rejected):
        """
        Compute a new value for the covariance sacling factor based on the acceptance/rejection
        ratio
        """
        # unpack my weights
        aw = self.acceptanceWeight
        rw = self.rejectionWeight
        # compute the acceptance ratio
        acceptance = accepted / (accepted + rejected + invalid)
        # the fudge factor
        kc = (aw*acceptance + rw)/(aw+rw)
        # don't let it get too small
        if kc < .1: kc = .1
        # or too big
        if kc > 1.: kc = 1.
        # store it
        self.scaling = kc

        # and return
        return self

    def allocateGPUData(self, samples, parameters):
        """
        initialize gpu work data
        """
        precision = self.precision
        # allocate  of cudaCoolingStep
        self.gcandidate = self.cudaCoolingStep.alloc(samples, parameters, dtype=precision)
        self.gproposal = altar.cuda.matrix(shape=(samples, parameters), dtype=precision)

        # allocate sigma_chol
        self.gsigma_chol = altar.cuda.matrix(shape=(parameters, parameters), dtype=precision)

        # allocate local
        self.ginvalid_flags = altar.cuda.vector(shape=samples, dtype='int32')
        self.gacceptance_flags = altar.cuda.vector(shape=samples, dtype='int32')
        self.gvalid_indices = altar.cuda.vector(shape=samples, dtype='int32')
        self.gvalid_samples = altar.cuda.vector(shape=1, dtype='int32')

        self.gdice = altar.cuda.vector(shape=samples, dtype=precision)

        # set initialized flag = 1
        self.ginit = True
        return

    # private data
    mcsteps = 1          # the length of each Markov chain
    dispatcher = None  # a reference to the event dispatcher
    ginit = False     # whether gpu data are allocated
    gstep = None # cuda/gpu step for keeping sampling states
    gcandidate = None # cuda/gpu candidate state
    gproposal = None # save theta jumps
    gsigma_chol = None  # placeholder for the scaled and decomposed parameter covariance matrix
    gvalid_indices = None
    gvalid_samples = None
    ginvalid_flags = None
    gacceptance_flags = None
    precision = None
    gdice = None
    curng = None

# end of file
