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
from __future__ import annotations
import typing
import numpy
# superclass
from .LangevinMethod import LangevinMethod

if typing.TYPE_CHECKING:
    from altar.bayesian.controllers.Langevin import Langevin
    from altar.shells.Application import Application


# declaration
class SequentialLangevin(LangevinMethod):
    """
    Implementation that assumes its state is the global state of the solver, and therefore it
    is able to compute the statistical properties of the sample distribution
    """


    # public data
    wid = 0     # my worker id
    workers: int = 1 # i don't manage anybody else


    # interface
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me and my parts given an {application} context
        """
        # chain up
        super().initialize(application=application)
        # grab the rng, for the SGLD noise term
        self.rng = application.rng.rng
        # all done
        return self


    def start(self, controller: Langevin) -> typing.Self:
        """
        Start the langevin process
        """
        # chain up
        super().start(controller=controller)
        # build a langevin step to hold the state of the problem
        self.step = self.LangevinStep.start(annealer=controller)
        # all done
        return self


    def walk(self, controller: Langevin) -> typing.Self:
        """
        SGLD walk: sweep {controller.sweeps} times, recomputing the gradients and taking an
        SGLD step at the current sampling rate {controller.epsilon_t} each time
        """
        # increment the iteration index
        self.iteration += 1

        # grab the state and set the sampling rate
        step = self.step
        step.epsilon_t = controller.epsilon_t

        # grab the model
        model = controller.model

        # iterate {sweeps} times for a given epsilon_t
        for sweep in range(controller.sweeps):
            # compute prior and data likelihood gradients, in sampling space
            step.compute_gradients(controller=controller)
            # update theta, and its physical values when reparameterized
            step.updateTheta(rng=self.rng)
            step.refresh_physical(model=model)

        # all done
        return self


    def rate_statistics(self, controller: Langevin
                        ) -> tuple[int, numpy.ndarray, numpy.ndarray, float]:
        """
        The statistics {estimate_rate} needs from my chains, in sampling space
        """
        step = self.step
        # the gradients, in sampling space
        step.compute_gradients(controller=controller)
        gradient = step.grad_prior + step.grad_data
        θ = step.theta_sampling
        # all done
        return θ.shape[0], θ.sum(axis=0), (θ * θ).sum(axis=0), numpy.abs(gradient).max()


    # private data
    rng: numpy.random.Generator


# end of file
