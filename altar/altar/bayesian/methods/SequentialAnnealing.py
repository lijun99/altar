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


# declaration
class SequentialAnnealing(AnnealingMethod):
    """
    Implementation that assumes its state is the global state of the solver, and therefore it
    is able to compute the statistical properties of the sample distribution
    """


    # public data
    wid = 0     # my worker id
    workers = 1 # i don't manage anybody else


    # interface
    def start(self, annealer):
        """
        Start the annealing process
        """
        # chain up
        super().start(annealer=annealer)
        # build a cooling step to hold the state of the problem
        self.step = self.CoolingStep.start(annealer=annealer)
        # notify the archiver
        annealer.archiver.start(step=self.step, iteration=self.iteration, psets=annealer.model.psets)
        # all done
        return self


    def restart(self, annealer, checkpoint, share=None):
        """
        Start the annealing process from a {checkpoint}, a step an earlier run archived
        """
        # chain up
        super().restart(annealer=annealer, checkpoint=checkpoint, share=share)
        # build a cooling step to hold the state of the problem
        model = annealer.model
        step = self.CoolingStep.allocate(annealer=annealer)
        # my rows of the population, and their data likelihoods, at the step's temperature
        total, start = share or (step.samples, 0)
        theta, data = checkpoint.rows(total=total, start=start, count=step.samples)
        step.beta = checkpoint.beta
        step.theta[...] = theta
        step.data[...] = data
        # the sampling space, the prior and the posterior, from the samples
        step.refresh_sampling(model=model)
        step.prior[...] = 0
        model.eval_prior(step=step)
        model.eval_posterior(step=step)
        self.step = step
        # notify the archiver
        annealer.archiver.start(step=step, iteration=self.iteration, psets=model.psets)
        # all done
        return self


# end of file
