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
import altar.cuda

# declaration
class CUDAAnnealing(AnnealingMethod):
    """
    Implementation that takes advantage of CUDA on gpus to accelerate the computation
    """

    from altar.bayesian.states.cuda.CoolingStep import CoolingStep as cudaCoolingStep

    # public data
    wid = 0     # my worker id
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
        gstep.copy_from_cpu(step=self.step)
        gstep.prior.zero(), gstep.data.zero(), gstep.posterior.zero()
        annealer.model.likelihoods(annealer=annealer, step=gstep, batch=gstep.samples)
        gstep.copy_to_cpu(step=self.step)
        return self


    def start(self, annealer):
        """
        Start the annealing process
        """
        # chain up
        super().start(annealer=annealer)
        # assign a cuda device to worker in sequence of the worker id
        # create both cpu/gpu steps to hold the state of the problem
        self.step = self.CoolingStep.allocate(annealer=annealer)
        self.gstep = self.cudaCoolingStep.start(annealer=annealer)

        # initialize it
        model = annealer.model
        gstep = self.gstep
        model.initialize_sample(step=gstep, batch=gstep.samples)
        # compute the likelihoods
        model.likelihoods(annealer=annealer, step=gstep, batch=gstep.samples)
        # log|J| of the initial sample, kept apart from the physical-space prior
        if gstep.has_reparametrization:
            gstep.jacobian.zero()
            model.eval_prior_with_physical(step=gstep, likelihood=gstep.jacobian, batch=gstep.samples)
        # return to cpu
        gstep.copy_to_cpu(step=self.step)

        # notify the archiver
        annealer.archiver.start(step=self.step, iteration=self.iteration, psets=annealer.model.psets)

        # all done
        return self

    device = None
    gstep = None

# end of file
