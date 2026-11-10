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
# my superclass
from .Annealer import Annealer
# my parts
from ..samplers.Perturbation import Perturbation as sampler
from ..schedulers.ConstantTemperature import ConstantTemperature


# declaration
class Perturbation(Annealer, family="altar.controllers.perturbation"):
    """
    Exact sampling of linear-gaussian posteriors: {Annealer} at β = 1, with its sampler pinned to
    {Perturbation}. Without bounds, one walk draws exact, independent samples; with bounds, the
    chains walk {rounds} times, {job.steps} exact trajectories each. The log evidence comes with
    the samples. With an adaptive C_p, re-estimated from the mean of the samples before each walk,
    the chains walk until it changes by less than {cp_tolerance}, up to {max_rounds} walks.
    """

    # user configurable state
    sampler = altar.bayesian.sampler(default=sampler)
    sampler.doc = "the sampler of the posterior distribution"

    scheduler = altar.bayesian.scheduler(default=ConstantTemperature)
    scheduler.doc = "keeps beta fixed at 1"

    rounds = altar.properties.int(default=1)
    rounds.validators = altar.constraints.isGreaterEqual(value=1)
    rounds.doc = "the walks; one is enough without bounds"

    cp_tolerance = altar.properties.float(default=0.01)
    cp_tolerance.validators = altar.constraints.isGreater(value=0)
    cp_tolerance.doc = "with an adaptive C_p, walk until it changes by less than this from one " \
                       "walk to the next, in the frobenius norm, relative"

    max_rounds = altar.properties.int(default=20)
    max_rounds.validators = altar.constraints.isGreaterEqual(value=1)
    max_rounds.doc = "the most walks, with an adaptive C_p"


    # implementation details
    def continuing(self, worker, iteration, tolerance):
        """
        Whether to walk the chains again, after {iteration} walks: {rounds} walks, and, with an
        adaptive C_p, until it settles
        """
        walks = iteration - self.start_iteration
        cp = getattr(getattr(self.model, "cp", None), "latest", None)
        previous, self._cp = self._cp, cp
        if walks < self.rounds:
            return True
        if cp is None or cp is previous:
            return False
        if previous is None:
            return walks < self.max_rounds
        change = float(numpy.linalg.norm(cp - previous) / numpy.linalg.norm(cp))
        self.info.log(f"perturbation: C_p changed by {change:.3g} in walk {walks}")
        if change < self.cp_tolerance:
            return False
        if walks >= self.max_rounds:
            self.warning.log(f"perturbation: C_p still changes by {change:.3g} after "
                             f"{walks} walks")
            return False
        return True


    # private data
    _cp = None # the C_p of the last walk


# end of file
