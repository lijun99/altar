# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
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
from altar.bayesian.samplers.Sampler import Sampler


# declaration
class Metropolis(altar.component, family="altar.samplers.metropolis", implements=Sampler):
    """
    The Metropolis algorithm as a sampler of the posterior distribution
    """

    # types
    from altar.bayesian.states.cuda.CoolingStep import CoolingStep


    # user configurable state
    scaling = altar.properties.float(default=.1)
    scaling.doc = 'the parameter covariance Σ is scaled by the square of this'

    acceptanceWeight = altar.properties.float(default=8.0/9.0)
    acceptanceWeight.doc = 'the weight of accepted samples during covariance rescaling'

    rejectionWeight = altar.properties.float(default=1.0/9.0)
    rejectionWeight.doc = 'the weight of rejected samples during covariance rescaling'

    useFixedScaling = altar.properties.bool(default=False)
    useFixedScaling.doc = 'whether to use a fixed scaling'

    scalingMin = altar.properties.float(default=.01)
    scalingMin.doc = 'the minimum value of the scaling factor'

    scalingMax = altar.properties.float(default=1)
    scalingMax.doc = 'the maximum value of the scaling factor'

    # proposal mechanism: cpu-only (its covariance computation runs against the cpu-side
    # step this sampler is handed in {prepare_sampling_pdf}); only its unscaled covariance
    # {_sigma} is used here, uploaded to the gpu and decomposed/scaled there (see
    # {prepare_sampling_pdf}), so there is no cuda counterpart needed for the covariance
    # computation itself -- it's a small (parameters x parameters) matrix, cheap on cpu
    proposal = altar.bayesian.proposal()
    proposal.doc = "the proposal mechanism used by this sampler"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        # pull the chain length from the job specification
        self.mcsteps = application.job.steps
        # the curand generator cached on the current device
        self.curng = altar.cuda.curand_generator()
        self.precision = application.job.gpuprecision
        # initialize the (cpu) proposal mechanism
        self.proposal.initialize(application=application)

        # all done
        return self

    def cu_initialize(self, application):
        self.initialize(application=application)
        return self

    @altar.export
    def sample_posterior(self, annealer, step):
        """
        Sample the posterior distribution
        Arguments:
            annealer - the controller
            step - cpu CoolingStep
        Return:
            statistics (accepted, invalid, rejected)
        """
        # grab the dispatcher
        dispatcher = annealer.dispatcher
        # notify we have started sampling the posterior
        dispatcher.notify(event=dispatcher.sample_posterior_start, controller=annealer)

        # prepare the sampling pdf, copy step to gpu step
        self.prepare_sampling_pdf(annealer=annealer, step=step)

        # check whether model parameters needed to be updated, e.g., Cp
        model = annealer.model

        if model.update_model(annealer=annealer):
            # if updated, recompute datalikelihood and posterior
            gstep = self.gstep
            batch = gstep.samples
            gstep.prior.zero(), gstep.data.zero(), gstep.posterior.zero()
            model.likelihoods(annealer=annealer, step=gstep, batch=batch)

        # walk the chains
        statistics = self.walk_chains(annealer=annealer, step=self.gstep)

        # finish the sampling pdf, copy gpu step back
        self.finish_sampling_pdf(step=step)

        # notify we are done sampling the posterior
        dispatcher.notify(event=dispatcher.sample_posterior_finish, controller=annealer)
        # all done
        return statistics


    @altar.export
    def update(self, annealer, statistics):
        """
        Update my parameters based on the results of walking my Markov chains
        """
        # update the scaling of the parameter covariance matrix
        if not self.useFixedScaling:
            self.adjust_covariance_scaling(*statistics)
        # all done
        return


    # implementation details
    def prepare_sampling_pdf(self, annealer, step):
        """
        Re-scale and decompose the parameter covariance matrix, in preparation for the
        Metropolis update
        """
        # get the dispatcher
        dispatcher = annealer.dispatcher
        # notify we have started preparing the sampling PDF
        dispatcher.notify(event=dispatcher.prepare_sampling_pdf_start, controller=annealer)

        # allocate local gpu data if not allocated
        self.gstep = annealer.worker.gstep
        if self.ginit is not True:
            self.allocate_gpu_data(step.samples, step.parameters)

        # copy cpu step state
        self.gstep.copy_from_cpu(step=step)

        # let the (cpu) proposal component (re-)compute the *unscaled* parameter covariance
        # Σ from the weighted sample auto-correlation, exactly as the cpu Metropolis sampler
        # does (same {step}, same weights) -- {_prepare} caches on beta/scaling, so this is a
        # cheap no-op except right after a new annealing step; only {_sigma} (unscaled) is
        # used, not {_sigma_chol} (whose triangle/decomposition convention isn't ours) --
        # decomposing on the gpu below and scaling the *factor* by {self.scaling} (not
        # squared) is mathematically identical to scaling Σ by scaling^2 before decomposing,
        # since (c·U)^T(c·U) = c^2 · U^T U
        self.proposal._prepare(sampler=self, step=step, annealer=annealer)
        self.gsigma_chol.copy_from_host(source=self.proposal._sigma)

        # compute its Cholesky decomposition: self = U^T U, U left in the row-major upper
        # triangle (see {altar.cuda.array.Array.cholesky}, the same convention
        # {altar.norms.cuda.L2}/{altar.data.cuda.DataL2} already validated against real gpu
        # output)
        self.gsigma_chol.cholesky()

        # scale it
        self.gsigma_chol *= self.scaling

        # notify we are done preparing the sampling PDF
        dispatcher.notify(event=dispatcher.prepare_sampling_pdf_finish, controller=annealer)
        # all done
        return

    def finish_sampling_pdf(self, step):
        """
        procedures after sampling, e.g, copy data back to cpu
        """
        # copy gpu step back to cpu
        self.gstep.copy_to_cpu(step=step)
        return

    def walk_chains(self, annealer, step):
        """
        Run the Metropolis algorithm on the Markov chains
        Arguments:
            annealer: cudaAnnealer
            step: CoolingStep
        Return:
            statistics = (accepted, invalid, rejected)
        """
        # get the model
        model = annealer.model
        # and the event dispatcher
        dispatcher = annealer.dispatcher
        # the metropolis extension bindings
        metropolis = libcudaaltar.metropolis

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

        # step all chains together
        for ihop in range(self.mcsteps):
            # notify we are advancing the chains
            dispatcher.notify(event=dispatcher.chain_advance_start, controller=annealer)

            # notify we are starting the verification process
            dispatcher.notify(event=dispatcher.verify_start, controller=annealer)


            # the random displacement may have generated candidates that are outside the
            # support of the model, so we must give it an opportunity to reject them;
            # initialize the candidate sample by randomly displacing the current one
            self.displace(displacement=θproposal)
            θproposal += θ

            # reset the mask and ask the model to verify the sample validity
            # note that I have redefined model.verify to use theta as input

            model.verify_theta(theta=θproposal, mask=invalid_flags.zero(), batch=samples)

            invalid_step = invalid_flags.sum()
            valid = samples - invalid_step
            # if valid = 0, continue to next MC step
            if valid == 0 :
                invalid += invalid_step
                continue

            # if valid > 0, proceed to Metropolis accept-reject
            # set indices for valid samples, return valid samples count
            metropolis.cudaMetropolis_setValidSampleIndices(
                valid_indices.grid, invalid_flags.grid, valid_samples.grid)

            # get the invalid samples count
            invalid += invalid_step

            # queue valid samples to first rows of cθ
            metropolis.cudaMetropolis_queueValidSamples(
                cθ.grid, θproposal.grid, valid_indices.grid, valid)

            # notify that the verification process is finished
            dispatcher.notify(event=dispatcher.verify_finish, controller=annealer)

            # initialize the likelihoods
            likelihoods = cprior.zero(), cdata.zero(), cpost.zero()

            # compute the probabilities/likelihoods
            model.likelihoods(annealer=annealer, step=candidate, batch=valid)

            # randomize the Metropolis acceptance vector
            curand.uniform(out=dice, generator=self.curng)

            # notify we are starting accepting samples
            dispatcher.notify(event=dispatcher.accept_start, controller=annealer)

            # accept/reject: go through all the samples
            metropolis.cudaMetropolis_metropolisUpdate(
                θ.grid, prior.grid, data.grid, posterior.grid,      # original
                cθ.grid, cprior.grid, cdata.grid, cpost.grid,       # candidate
                dice.grid, acceptance_flags.zero().grid, valid_indices.grid, valid)

            # counting the acceptance/rejection
            accepted_step = acceptance_flags.sum()
            accepted += accepted_step
            rejected += valid - accepted_step

            # notify we are done accepting samples
            dispatcher.notify(event=dispatcher.accept_finish, controller=annealer)

            # notify we are done advancing the chains
            dispatcher.notify(event=dispatcher.chain_advance_finish, controller=annealer)
        # all done
        return accepted, invalid, rejected


    def displace(self, displacement):
        """
        Construct a set of displacement vectors for the random walk from a distribution with zero
        mean and my covariance
        """
        # get my decomposed covariance: Σ = U^T U, U in the row-major upper triangle
        Σ_chol = self.gsigma_chol
        samples, parameters = displacement.shape

        # generate gaussian random numbers (samples x parameters)
        curand.gaussian(out=displacement, generator=self.curng)

        # displacement <- displacement @ U, the same row-major "swap trick" already validated
        # in {altar.norms.cuda.L2._apply_covariance}: the triangular operand ({sigma_chol})
        # is always cublas's "A", only the side/uplo flags swap for the row-major right-
        # multiply, and reading {sigma_chol}'s row-major upper triangle as column-major
        # lower is what makes {uplo=LOWER} (not UPPER) the correct flag here
        double = displacement.dtype == "float64"
        trmm = cublas.dtrmm if double else cublas.strmm
        trmm(
            altar.cuda.cublas_handle(),
            cublas.SideMode.LEFT, cublas.FillMode.LOWER, cublas.Operation.N,
            cublas.DiagType.NON_UNIT,
            parameters, samples, 1.0,
            Σ_chol.grid, parameters,
            displacement.grid, parameters,
            displacement.grid, parameters,
        )
        # and return
        return displacement


    def adjust_covariance_scaling(self, accepted, invalid, rejected):
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
        kc = aw*acceptance + rw
        # don't let it get too small
        kc = max(kc, self.scalingMin)
        # or too big
        kc = min(kc, self.scalingMax)
        # store it
        self.scaling = kc

        # and return
        return self

    def allocate_gpu_data(self, samples, parameters):
        """
        initialize gpu work data
        """
        precision = self.precision
        # allocate a CoolingStep
        self.gcandidate = self.CoolingStep.alloc(samples, parameters, dtype=precision)
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
