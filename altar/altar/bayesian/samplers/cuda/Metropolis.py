# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import math
# the package
import altar
import altar.cuda
from altar.cuda import curand
from altar.cuda import cublas
from altar.cuda import libcudaaltar


# declaration
class Metropolis:
    """
    The cuda implementation of the Metropolis algorithm as a sampler of the posterior
    distribution. See {altar.bayesian.samplers.Metropolis}, the pyre component a {.pfg}
    actually selects, which picks me (or my cpu counterpart) once, at {initialize} time.
    """

    # types
    from altar.bayesian.states.cuda.CoolingStep import CoolingStep


    # protocol-shaped obligations (called by the shim, not pyre-dispatched directly)
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        # the curand generator cached on the current device
        self.curng = altar.cuda.curand_generator()
        self.precision = application.job.gpuprecision
        # random-walk Metropolis's theoretically-optimal acceptance rate in high dimensions
        # (Roberts-Gelman-Gilks), unless the user picked a target explicitly
        if getattr(self.stepsizer, "target", None) is None:
            self.stepsizer.target = 0.234
        # initialize the step size regulator and record the initial scaling
        self.scaling = self.stepsizer.initialize(value=self.scaling)
        # initialize the step count regulator (e.g. {FixedSteps} fills in application.job.steps)
        self.stepcounter.initialize(application=application)
        # initialize the (cpu) proposal mechanism
        self.proposal.initialize(application=application)

        # all done
        return self

    def cu_initialize(self, application):
        self.initialize(application=application)
        return self

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

        # walk the chains
        statistics = self.walk_chains(annealer=annealer, step=self.gstep)

        # finish the sampling pdf, copy gpu step back
        self.finish_sampling_pdf(step=step)

        # notify we are done sampling the posterior
        dispatcher.notify(event=dispatcher.sample_posterior_finish, controller=annealer)
        # all done
        return statistics


    def update(self, annealer, statistics):
        """
        Update my parameters based on the results of walking my Markov chains
        """
        # unpack the statistics
        accepted, invalid, rejected = statistics
        # delegate step size adjustment to the stepsizer
        self.scaling = self.stepsizer.adjust(
            attempts=accepted + invalid + rejected,
            accepted=accepted)
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
            self.allocate_gpu_data(step.samples, step.parameters,
                reparameterized=step.has_reparametrization)

        # copy cpu step state
        self.gstep.copy_from_cpu(step=step)

        # let the (cpu) proposal component (re-)compute the *unscaled* parameter covariance
        # Σ from the weighted sample auto-correlation, exactly as the cpu Metropolis sampler
        # does (same {step}, same weights), recomputing it at the start of each walk; only
        # {_sigma} (unscaled) is used, not {_sigma_chol} (whose triangle/decomposition convention isn't ours) --
        # decomposing on the gpu below and scaling the *factor* by {self.scaling} (not
        # squared) is mathematically identical to scaling Σ by scaling^2 before decomposing,
        # since (c·U)^T(c·U) = c^2 · U^T U
        # a reparameterized model walks in sampling space, so Σ comes from those samples
        walker = step
        if step.has_reparametrization:
            from altar.bayesian.states.CoolingStep import CoolingStep
            walker = CoolingStep(beta=step.beta, theta=step.theta_sampling,
                likelihoods=(step.prior, step.data, step.posterior))
            walker.weights = getattr(step, "weights", None)
        self.proposal.new_walk()
        self.proposal._prepare(sampler=self, step=walker, annealer=annealer)
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

        # a reparameterized model walks in sampling space, where every proposal is in the
        # support; the target there is the posterior plus the log-jacobian of the map, which
        # {posterior} carries for the length of the walk
        reparameterized = step.has_reparametrization
        if reparameterized:
            θs = step.theta_sampling
            jacobian = step.jacobian
            model.eval_prior_with_physical(step=step, likelihood=jacobian.zero(), batch=samples)
            posterior += jacobian
            θphysical = self.gproposal_physical
            cθs = self.gcandidate_sampling
            cjacobian = self.gcandidate_jacobian

        # the step count regulator decides how many MC steps to run, in blocks, before
        # checking whether this β step is done ({FixedSteps}: one block, the fixed count;
        # {DecorrelatingSteps}: repeated blocks until decorrelated)
        self.stepcounter.start(theta=θ, beta=β)
        mcsteps = 0

        while not self.stepcounter.done(mcsteps=mcsteps, theta=θ, annealer=annealer):
            block = self.stepcounter.block_size()
            for ihop in range(block):
                # notify we are advancing the chains
                dispatcher.notify(event=dispatcher.chain_advance_start, controller=annealer)

                # notify we are starting the verification process
                dispatcher.notify(event=dispatcher.verify_start, controller=annealer)


                # the random displacement may have generated candidates that are outside the
                # support of the model, so we must give it an opportunity to reject them;
                # initialize the candidate sample by randomly displacing the current one
                self.displace(displacement=θproposal)
                θproposal += θs if reparameterized else θ
                # in sampling space, map a copy of the proposal to physical
                if reparameterized:
                    θphysical.copy(θproposal)
                    model.to_physical(theta=θphysical, batch=samples)
                else:
                    θphysical = θproposal

                # reset the mask and ask the model to verify the sample validity
                # note that I have redefined model.verify to use theta as input

                model.verify_theta(theta=θphysical, mask=invalid_flags.zero(), batch=samples)

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
                    cθ.grid, θphysical.grid, valid_indices.grid, valid)
                if reparameterized:
                    metropolis.cudaMetropolis_queueValidSamples(
                        cθs.grid, θproposal.grid, valid_indices.grid, valid)

                # notify that the verification process is finished
                dispatcher.notify(event=dispatcher.verify_finish, controller=annealer)

                # initialize the likelihoods
                likelihoods = cprior.zero(), cdata.zero(), cpost.zero()

                # compute the probabilities/likelihoods
                model.likelihoods(annealer=annealer, step=candidate, batch=valid)
                # in sampling space, add the log-jacobian of the candidates
                if reparameterized:
                    model.eval_prior_with_physical(
                        step=candidate, likelihood=cjacobian.zero(), batch=valid)
                    cpost += cjacobian

                # randomize the Metropolis acceptance vector
                curand.uniform(out=dice, generator=self.curng)

                # notify we are starting accepting samples
                dispatcher.notify(event=dispatcher.accept_start, controller=annealer)

                # accept/reject: go through all the samples
                metropolis.cudaMetropolis_metropolisUpdate(
                    θ.grid, prior.grid, data.grid, posterior.grid,      # original
                    cθ.grid, cprior.grid, cdata.grid, cpost.grid,       # candidate
                    dice.grid, acceptance_flags.zero().grid, valid_indices.grid, valid)
                # and bring the sampling space of the accepted candidates along
                if reparameterized:
                    metropolis.cudaMetropolis_updateSampling(
                        θs.grid, jacobian.grid, cθs.grid, cjacobian.grid,
                        acceptance_flags.grid, valid_indices.grid, valid)

                # counting the acceptance/rejection
                accepted_step = acceptance_flags.sum()
                accepted += accepted_step
                rejected += valid - accepted_step

                # notify we are done accepting samples
                dispatcher.notify(event=dispatcher.accept_finish, controller=annealer)

                # notify we are done advancing the chains
                dispatcher.notify(event=dispatcher.chain_advance_finish, controller=annealer)

            mcsteps += block
        # leave the posterior itself in {posterior}
        if reparameterized:
            posterior -= jacobian
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


    def allocate_gpu_data(self, samples, parameters, reparameterized=False):
        """
        initialize gpu work data
        """
        precision = self.precision
        # allocate a CoolingStep
        self.gcandidate = self.CoolingStep.alloc(samples, parameters, dtype=precision)
        self.gproposal = altar.cuda.matrix(shape=(samples, parameters), dtype=precision)
        # a walk in sampling space keeps the proposals in physical space, and the candidates'
        # sampling space and log-jacobian, apart
        if reparameterized:
            self.gproposal_physical = altar.cuda.matrix(shape=(samples, parameters), dtype=precision)
            self.gcandidate_sampling = altar.cuda.matrix(shape=(samples, parameters), dtype=precision)
            self.gcandidate_jacobian = altar.cuda.vector(shape=samples, dtype=precision)

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

    # private data; the component-typed attributes are set by the shim's initialize()
    # before it calls mine (see {altar.bayesian.samplers.Metropolis._makeImpl})
    proposal = None    # the proposal mechanism used by this sampler
    stepsizer = None   # the step size regulator
    stepcounter = None # the step count regulator
    scaling = 0.1      # the parameter covariance Σ is scaled by the square of this

    dispatcher = None  # a reference to the event dispatcher
    ginit = False     # whether gpu data are allocated
    gstep = None # cuda/gpu step for keeping sampling states
    gcandidate = None # cuda/gpu candidate state
    gproposal = None # save theta jumps
    gproposal_physical = None # the jumps in physical space, when walking in sampling space
    gcandidate_sampling = None # the candidates in sampling space
    gcandidate_jacobian = None # the log-jacobian of the candidates
    gsigma_chol = None  # placeholder for the scaled and decomposed parameter covariance matrix
    gvalid_indices = None
    gvalid_samples = None
    ginvalid_flags = None
    gacceptance_flags = None
    precision = None
    gdice = None
    curng = None

# end of file
