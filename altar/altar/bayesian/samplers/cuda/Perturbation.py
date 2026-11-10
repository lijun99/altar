# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


from ..native.Perturbation import Perturbation as native


class Perturbation(native):
    """
    The cuda implementation of the perturbation sampler: the draws on the host, in double
    precision, the likelihoods on the device; see {altar.bayesian.samplers.Perturbation}
    """

    # backend hooks
    def store(self, annealer, step, theta):
        """
        The likelihoods of the samples of the cpu {step}, on the device, a batch of chains at a
        time
        """
        model = annealer.model
        gstep = annealer.worker.gstep
        for offset in range(0, step.samples, gstep.samples):
            gstep.copy_from_cpu(step=step, offset=offset)
            gstep.refresh_sampling(model=model, batch=gstep.samples)
            gstep.prior.zero(), gstep.data.zero(), gstep.posterior.zero()
            model.likelihoods(annealer=annealer, step=gstep, batch=gstep.samples)
            gstep.copy_to_cpu(step=step, offset=offset)
        return


# end of file
