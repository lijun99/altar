# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# superclass
from .AnnealingMethod import AnnealingMethod
from .Pool import Pool
import altar.cuda

# declaration
class CUDAAnnealing(AnnealingMethod):
    """
    Implementation that takes advantage of CUDA on gpus to accelerate the computation
    """

    from altar.bayesian.states.cuda.CoolingStep import CoolingStep as cudaCoolingStep

    # public data
    wid = 0     # my worker id
    pools = True # i keep a pool of the states of the chains
    workers = 1 # i don't manage anybody else

    def initialize(self, application):
        """
        initialize worker
        """
        super().initialize(application=application)
        # ensure cuda backend is active
        altar.backends.activate_cuda()
        self.cu_initialize(application=application)
        # all done
        return self

    def cu_initialize(self, application):
        """
        Initialize the cuda worker
        """
        gpuids = application.job.gpuids
        tasks = application.job.tasks # jobs per host
        # set gpu ids for current worker
        did = gpuids[self.wid % tasks]
        altar.cuda.manager.device(did)
        self.device = altar.cuda.manager.devices[did]
        application.info.log(f'current worker {self.wid} with device {self.device.name} id {self.device.id}')
        return self

    # interface
    def densities(self, annealer):
        """
        Recompute the densities of my step on the gpu, after the model changed
        """
        gstep = self.gstep
        # the population, a pool slot at a time
        for offset in range(0, self.step.samples, gstep.samples):
            gstep.copy_from_cpu(step=self.step, offset=offset)
            gstep.prior.zero(), gstep.data.zero(), gstep.posterior.zero()
            annealer.model.likelihoods(annealer=annealer, step=gstep, batch=gstep.samples)
            gstep.copy_to_cpu(step=self.step, offset=offset)
        return self


    def start(self, annealer):
        """
        Start the annealing process
        """
        # chain up
        super().start(annealer=annealer)
        # my cpu and gpu steps
        reparameterized = self._allocate(annealer=annealer)
        model = annealer.model
        chains = model.job.chains

        # draw the initial population at once, so that preset samples don't repeat
        gstep = self.gstep
        population = gstep if self.pool is None else self.cudaCoolingStep.alloc(
            samples=self.step.samples, parameters=model.parameters,
            dtype=model.job.gpuprecision, has_reparametrization=reparameterized)
        model.initialize_sample(step=population, batch=population.samples)
        population.copy_to_cpu(step=self.step)
        # and compute its densities, the chains at a time
        for offset in range(0, self.step.samples, chains):
            gstep.copy_from_cpu(step=self.step, offset=offset)
            model.likelihoods(annealer=annealer, step=gstep, batch=gstep.samples)
            # log|J| of the initial sample, kept apart from the physical-space prior
            if gstep.has_reparametrization:
                gstep.jacobian.zero()
                model.eval_prior_with_physical(step=gstep, likelihood=gstep.jacobian, batch=gstep.samples)
            gstep.copy_to_cpu(step=self.step, offset=offset)

        # notify the archiver
        annealer.archiver.start(step=self.step, iteration=self.iteration, psets=getattr(annealer.model, "psets", None))

        # all done
        return self

    def restart(self, annealer, checkpoint, share=None):
        """
        Start the annealing process from a {checkpoint}, a step an earlier run archived
        """
        # chain up
        super().restart(annealer=annealer, checkpoint=checkpoint, share=share)
        # my cpu and gpu steps
        self._allocate(annealer=annealer)
        model = annealer.model
        step, gstep = self.step, self.gstep
        # my rows of the population, and their data likelihoods, at the step's temperature
        total, start = share or (step.samples, 0)
        theta, data = checkpoint.rows(total=total, start=start, count=step.samples)
        step.beta = checkpoint.beta
        step.theta[...] = theta
        step.data[...] = data
        # the rest of the densities, the chains at a time: the prior and the sampling space
        # from the samples, and the posterior from the prior and the data likelihood
        for offset in range(0, step.samples, gstep.samples):
            gstep.copy_from_cpu(step=step, offset=offset)
            # the data likelihoods of the current model, the way the walk computes them
            if annealer.restart_recompute:
                gstep.prior.zero(); gstep.data.zero(); gstep.posterior.zero()
                model.likelihoods(annealer=annealer, step=gstep, batch=gstep.samples)
            if gstep.has_reparametrization:
                gstep.theta_sampling.copy(gstep.theta)
                model.to_sampling(theta=gstep.theta_sampling, batch=gstep.samples)
                gstep.jacobian.zero()
                model.eval_prior_with_physical(step=gstep, likelihood=gstep.jacobian, batch=gstep.samples)
            gstep.prior.zero()
            model.eval_prior(step=gstep, batch=gstep.samples)
            model.eval_posterior(step=gstep, batch=gstep.samples)
            gstep.copy_to_cpu(step=step, offset=offset)
        # how far the data likelihoods moved, when recomputed
        if annealer.restart_recompute:
            self.compare(annealer=annealer, recorded=data, data=step.data)
        # notify the archiver
        annealer.archiver.start(step=step, iteration=self.iteration, psets=model.psets)
        # all done
        return self


    # implementation details
    def _allocate(self, annealer):
        """
        Build my cpu step, which holds the population, the states each chain keeps, and my gpu
        step, which holds the chains; return whether the model is reparameterized
        """
        model = annealer.model
        chains = model.job.chains
        # no pool keeps the chains' final states only, as without pooling
        size = annealer.pool
        self.pool = Pool(size=size, interval=annealer.pool_interval, chains=chains) if size > 1 else None
        reparameterized = getattr(model, 'has_reparametrization', False)
        self.step = self.CoolingStep.alloc(samples=chains * size,
            parameters=model.parameters, has_reparametrization=reparameterized)
        self.gstep = self.cudaCoolingStep.start(annealer=annealer)
        # all done
        return reparameterized


    device = None
    gstep = None
    pool = None # the states the chains keep while they walk; none without pooling

# end of file
