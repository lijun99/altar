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
    from altar.cuda.bayesian.cudaLangevinStep import cudaLangevinStep

    # public data
    step = None # the current state of the solver
    gstep = None # the gpu copy
    iteration = 0 # my iteration counter
    wid = 0     # my worker id
    workers = 1 # i don't manage anybody else
    device = None

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
        self.gstep = self.cudaLangevinStep.start(controller=controller)

        # initialize it
        model = controller.model
        gstep = self.gstep
        model.cuInitSample(theta=gstep.theta, batch=gstep.samples)
        # compute the likelihoods
        # model.likelihoods(controller=controller, step=gstep, batch=gstep.samples)
        # return to cpu
        gstep.copyToCPU(step=self.step)

        # all done
        return self

    def walk(self, controller):
        """
        SGLD walk
        """
        # increment the iteration index
        self.iteration += 1

        # grab the state
        step = self.gstep
        # get and set the sampling rate
        step.epsilon_t = controller.epsilon_t

        # grab the model
        model = controller.model

        # iterate {sweep} times for a given epsilon_t
        for sweep in range(controller.sweeps):

            # compute prior and data likelihood gradients
            model.gradient(controller=controller, step=step,  batch=step.samples)
            # update theta
            step.updateTheta()

        # all done
        return self

    def estimate_rate(self, controller, scale=1.0):
        """
        Estimate the sampling rate from std and gradient
        """

        # grab the state
        step = self.gstep

        # grab the model
        model = controller.model

        # compute the gradient
        model.gradient(controller=controller, step=step, batch=step.samples)
        gradient = step.data_gradient
        gradient += step.prior_gradient

        max_gradient = max(gradient.amax(), abs(gradient.amin()))

        mean, std = step.theta.mean_sd()
        max_std = std.amax()

        rate = scale*min(4*max_std/max_gradient, max_std*max_std)

        # all done
        return rate


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
        # compute the bayesian posterior
        controller.model.likelihoods(annealer=controller, step=self.gstep)
        # save them
        self.gstep.save_hdf5()

        # all done
        return self

    # meta-methods
    def __init__(self, controller, **kwds):
        # chain up; absorb the {controller}
        super().__init__(**kwds)
        # all done
        return


# end of file
