# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
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

    from altar.cuda.bayesian.cudaCoolingStep import cudaCoolingStep

    # public data
    wid = 0     # my worker id
    workers = 1 # i don't manage anybody else

    def initialize(self, application):
        """
        initialize worker
        """
        super().initialize(application=application)
        self.cuInitialize(application=application)
        # all done
        return self

    def cuInitialize(self, application):
        """
        Initialize the cuda worker
        """
        gpuids = application.job.gpuids
        tasks = application.job.tasks # jobs per host
        # set gpu ids for current worker
        self.device=altar.cuda.use_device(gpuids[self.wid % tasks])
        print(f'current worker {self.wid} with device {self.device} id {self.device.id}')
        return self

    # interface
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
        model.cuInitSample(theta=gstep.theta)
        # compute the likelihoods
        model.likelihoods(annealer=annealer, step=gstep, batch=gstep.samples)
        # return to cpu
        gstep.copyToCPU(step=self.step)

        # all done
        return self

    device = None
    gstep = None

# end of file
