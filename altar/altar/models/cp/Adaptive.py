# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import numpy
# the package
import altar
# my protocol
from .Cp import Cp


# declaration
class Adaptive(altar.component, family="altar.models.cp.adaptive", implements=Cp):
    """
    A C_p the model re-estimates at each beta step from the current posterior mean, or from an
    initial model early on; how C_p follows from a mean model is up to the model's {compute_cp}
    """


    # user configurable state
    start = altar.properties.float(default=0)
    start.doc = "include C_p once beta reaches this value"

    initial_model = altar.properties.path(default=None)
    initial_model.doc = "a mean model, in physical space, among the model's input files"

    initial_until = altar.properties.float(default=0)
    initial_until.doc = "use {initial_model} instead of the posterior mean while beta <= this"


    @altar.export
    def initialize(self, model, application):
        """
        Load the initial model, if any
        """
        if self.initial_model is not None:
            self._initial = numpy.asarray(
                model.io.load(filename=self.initial_model, shape=model.parameters, dtype="float64"))
        return self


    @altar.export
    def update(self, model, annealer, step):
        """
        Re-estimate C_p from a mean model and update the model's C_chi
        """
        beta = step.beta
        if beta < self.start:
            return False
        initial = self._initial is not None and beta <= self.initial_until
        mean = self._initial if initial else self.posterior_mean(annealer=annealer, step=step)
        cp = model.compute_cp(theta=mean)
        model.update_covariance(cp=cp)
        annealer.info.log(f"C_p from the {'initial' if initial else 'posterior'} mean model, "
                          f"at beta {beta:.6g}: trace {numpy.trace(cp):.6g}")
        return True


    @altar.export
    def apply(self, model, theta):
        """
        Estimate C_p from the mean model {theta} and update the model's C_chi
        """
        model.update_covariance(cp=model.compute_cp(theta=theta))
        return self


    def posterior_mean(self, annealer, step):
        """
        The mean of the physical samples in {step}, over all workers when running under mpi
        """
        θ = numpy.asarray(step.theta)
        total, count = θ.sum(axis=0), θ.shape[0]
        communicator = getattr(annealer.worker, "communicator", None)
        if communicator is not None and communicator.size > 1:
            total = numpy.array([communicator.sum(item=float(value)) for value in total])
            count = communicator.sum(item=count)
        return total / count


    # private data
    _initial = None


# end of file
