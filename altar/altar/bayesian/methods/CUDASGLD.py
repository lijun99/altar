# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# dependencies
import altar.cuda
from altar.cuda import libcudaaltar
# externals
import math
import numpy

# declaration
class CUDASGLD:
    """
    Implementation on Stochastic gradient Langevin dynamics (SGLD) with GPU
    """

    # classes to save step data
    from ..states.CoolingStep import CoolingStep
    from altar.bayesian.states.cuda.LangevinStep import LangevinStep as cudaLangevinStep

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
        model.initialize_sample(step=gstep, batch=gstep.samples)
        # compute the likelihoods
        # model.likelihoods(controller=controller, step=gstep, batch=gstep.samples)
        # return to cpu
        gstep.copy_to_cpu(step=self.step)

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

            # {model.gradient} always operates on physical-space theta (the forward model
            # needs real parameter values); refresh it from the sampling-space theta the
            # dynamics actually evolve, exactly as cuda {HMC} refreshes its own physical
            # snapshot after every leapfrog position update
            if step.has_reparametrization:
                step.theta.copy(step.theta_sampling)
                model.to_physical(theta=step.theta, batch=step.samples)

            # compute prior and data likelihood gradients
            model.gradient(controller=controller, step=step,  batch=step.samples)

            # {data_gradient} is w.r.t. physical theta; the chain rule needs it scaled by
            # d(physical)/d(sampling) to become a gradient w.r.t. sampling-space theta.
            # {prior_gradient} needs no such scaling: {Uniform.prior_gradient} (via its
            # {transform.jacobian_gradient}) already computes it directly in sampling space
            if step.has_reparametrization:
                model.eval_jacobian(step=step, batch=step.samples)
                numpy.asarray(step.data_gradient)[:] *= numpy.asarray(step.Jacobian)

            # update theta_sampling
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

        # refresh the physical-space snapshot, same as {walk}
        if step.has_reparametrization:
            step.theta.copy(step.theta_sampling)
            model.to_physical(theta=step.theta, batch=step.samples)

        # compute the gradient
        model.gradient(controller=controller, step=step, batch=step.samples)

        # scale {data_gradient} by the chain rule, same as {walk}
        if step.has_reparametrization:
            model.eval_jacobian(step=step, batch=step.samples)
            numpy.asarray(step.data_gradient)[:] *= numpy.asarray(step.Jacobian)

        gradient = step.data_gradient
        gradient += step.prior_gradient

        max_gradient = max(gradient.amax(), abs(gradient.amin()))

        mean, std = step.theta_sampling.mean_sd()
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
            self.gstep.report(controller=controller)

        # all done
        return self

    def finish(self, controller):
        """
        Procedures when simulation finishes
        """
        self.gstep.report(controller=controller)
        # all done
        return self

    # meta-methods
    def __init__(self, controller, **kwds):
        # chain up; absorb the {controller}
        super().__init__(**kwds)
        # all done
        return


# end of file
