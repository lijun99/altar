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
# my base class
from .Base import Base


# declaration
class Grid(Base, family="altar.bayesian.solvers.grid"):
    """
    A δβ solver based on an iterative grid search: scan ten steps across the interval that
    holds the answer, then refine within the step that crossed the target, ten times over
    """


    # implementation details
    def dbeta(self, high: float) -> float:
        """
        The δβ in [0, {high}] whose COV is my target
        """
        target, tolerance = self.target, self.tolerance
        low, bins = 0.0, 10
        for refinement in range(bins + 1):
            step = (high - low) / bins
            guess = low
            for scan in range(bins + 1):
                cov = self.cov_at(dbeta=guess)
                # close enough, or past the target on the very first point
                if abs(cov - target) < tolerance or (cov >= target and scan == 0):
                    return guess
                # past the target: refine in the step that crossed it
                if cov >= target:
                    break
                guess += step
            else:
                # the target is past the scanned interval: its upper end
                guess -= step
            low, high = guess - step, guess
        return guess


# end of file
