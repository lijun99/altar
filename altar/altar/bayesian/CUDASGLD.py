# -*- python -*-
# -*- coding: utf-8 -*-
#
# Lijun Zhu
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#


# dependencies
import altar.cuda
from altar.cuda import libcudaaltar
# externals
import math

# declaration
class CUDASGLD:
    """
    Implementation on Stochastic gradient Langevin dynamics (SGLD) with GPU
    """

    # classes to save step data
    from .CoolingStep import CoolingStep
    from altar.cuda.bayesian.cudaCoolingStep import cudaCoolingStep

    # public data
    step = None # the current state of the solver
    gstep = None # the gpu copy
    iteration = 0 # my iteration counter
    wid = 0     # my worker id
    workers = 1 # i don't manage anybody else
    device = None

    # private data
    eta_t = None
    theta_copy = None

    def initialize(self, application):
        """
        initialize worker
        """
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
        application.info.log(f'current worker {self.wid} with device {self.device} id {self.device.id}')

        samples = application.job.chains
        precision = application.job.gpuprecision
        eta = altar.cuda.vector(shape=samples, dtype=precision)

        # all done
        return self

    # interface
    def start(self, controller):
        """
        Start the annealing process
        """
        # chain up
        # super().start(controller=controller)
        # assign a cuda device to worker in sequence of the worker id
        # create both cpu/gpu steps to hold the state of the problem
        self.step = self.CoolingStep.allocate(annealer=controller)
        self.gstep = self.cudaCoolingStep.start(annealer=controller)

        # initialize it
        model = controller.model
        gstep = self.gstep
        model.cuInitSample(theta=gstep.theta, batch=gstep.samples)
        # compute the likelihoods
        # model.likelihoods(controller=controller, step=gstep, batch=gstep.samples)
        # return to cpu
        gstep.copyToCPU(step=self.step)

        self.eta_t = altar.cuda.vector(shape=gstep.samples, dtype=model.job.gpuprecision)

        # all done
        return self

    def walk(self, controller):
        """
        SGLD walk
        """

        self.iteration += 1

        # get the step size
        epsilon_t = controller.epsilon_t
        half_epsilon_t = epsilon_t/2.0
        sqrt_epsilon_t = math.sqrt(epsilon_t)

        model = controller.model
        parameters = model.parameters

        step = self.gstep
        #
        for sweep in range(controller.sweeps):
            # iterative over parameter
            for p in range(parameters):
                # compute prior and data likelihood gradients
                model.gradient(controller=controller, step=step, index=p, batch=step.samples)
                # generate gaussian random numbers (samples x parameters)
                altar.cuda.curand.gaussian(out=self.eta_t, scale=sqrt_epsilon_t)
                # theta += epsilon_t/2(prior_graident + data_gradient) + eta_t
                libcudaaltar.cudaLangevin_updateTheta(step.theta.data, step.prior.data, step.data.data,
                                                      half_epsilon_t, self.eta_t.data, p)


        # all done
        return self


    def bottom(self, controller):
        """
        Notification that we are at the bottom of an update
        """
        # notify the model
        controller.model.bottom(annealer=controller)

        if self.wid == 0 and self.iteration % controller.tsteps_report ==0:
            # get the state of the solution
            self.gstep.copyToCPU(step=self.step)
            step = self.step
            step.beta = controller.epsilon_t
            # calculate the statistics of samples
            step.statistics()
            # print a summary of current state
            step.print(channel=controller.info)

        # all done
        return self

    def finish(self, controller):
        """
        Procedures when simulation finishes
        """
        # all done
        return self

    # meta-methods
    def __init__(self, controller, **kwds):
        # chain up; absorb the {controller}
        super().__init__(**kwds)
        # all done
        return


# end of file
