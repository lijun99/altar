# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved
#
# Author(s): Codex

"""
Protocol and implementations for regulating sampler step sizes based on
acceptance statistics.
"""

import math

import altar


class StepSizer(altar.protocol, family="altar.bayesian.stepsizers"):
    """
    Protocol for components that adjust sampler step sizes.
    """

    @altar.provides
    def initialize(self, value=None):
        """
        Initialize internal state given an initial step size and return the value that should be used.
        """

    @altar.provides
    def adjust(self, *, attempts, accepted):
        """
        Return an updated step size based on an acceptance ratio.
        """

    @classmethod
    def pyre_default(cls, **kwds):
        """
        Default adjuster is the fixed strategy.
        """
        return FixedStepSize


class FixedStepSize(altar.component, family="altar.bayesian.stepsizers.fixedstep", implements=StepSizer):
    """
    Keep the step size fixed to a configured value.
    """

    step_size = altar.properties.float(default=0.01)
    step_size.doc = "the constant step size to use"

    @altar.export
    def initialize(self, value=None):
        if value is not None:
            self.step_size = value
        return self.step_size

    @altar.export
    def adjust(self, *, attempts, accepted):
        return self.step_size



class AdaptiveStepSizer(altar.component, family="altar.bayesian.stepsizers.adaptive", implements=StepSizer):
    """
    Base class that accumulates acceptance statistics and applies the
    concrete adjuster when the configured window is reached.
    """

    # configurable properties
    step_size = altar.properties.float(default=0.01)
    step_size.doc = "initial step size"

    step_window = altar.properties.int(default=1)
    step_window.doc = "number of proposals to accumulate before adjusting the step size; <=0 disables adaptation"

    min_step_size = altar.properties.float(default=1e-4)
    min_step_size.doc = "lower bound for the step size"

    max_step_size = altar.properties.float(default=1.0)
    max_step_size.doc = "upper bound for the step size"

    # internal state
    _attempts_total = 0
    _accepted_total = 0
    _attempts_since_adjust = 0
    _accepted_since_adjust = 0
    _iteration = 0

    @altar.export
    def initialize(self, value):
        self._reset_counters()
        return self._clip(value)

    @altar.export
    def adjust(self, *, attempts, accepted):
        """
        Concrete subclasses must implement this method to adjust the step size
        based on the acceptance ratio over the last window.
        """

        # record statistics
        self._record(attempts=attempts, accepted=accepted)

        # increment iteration counter
        self._iteration += 1

        if self._iteration >= self.step_window:
            # adaptation triggered; compute new step size
            self.step_size = self._clip(
                self.recompute_step_size())
            self._reset_window()

        # all done
        return self.step_size

    def recompute_step_size(self):
        """
        Recompute the step size based on accumulated statistics.
        """
        raise NotImplementedError("subclasses must implement recompute_step_size()")

    def _record(self, *, attempts, accepted):
        # accumulate statistics
        self._attempts_total += attempts
        self._accepted_total += accepted
        self._attempts_since_adjust += attempts
        self._accepted_since_adjust += accepted
        return

    def _reset_window(self):
        self._attempts_since_adjust = 0
        self._accepted_since_adjust = 0
        self._iteration = 0

    def _reset_counters(self):
        self._attempts_total = 0
        self._accepted_total = 0
        self._iteration = 0
        self._reset_window()

    def _clip(self, step):
        return max(self.min_step_size, min(self.max_step_size, float(step)))


class LinearRate(AdaptiveStepSizer, family="altar.bayesian.stepsizers.linearrate"):
    """
    Linear feedback strategy: step_size = intercept + slope * acceptance_ratio
    """

    intercept = altar.properties.float(default=0.01)
    intercept.doc = "constant term of the linear rule"

    slope = altar.properties.float(default=0.05)
    slope.doc = "coefficient multiplying the acceptance ratio"

    def recompute_step_size(self):
        # compute acceptance ratio
        ratio = (self._accepted_since_adjust / self._attempts_since_adjust
                 if self._attempts_since_adjust > 0 else 0.0)
        # return updated step size
        return self.intercept + self.slope * ratio


class TargetedRate(AdaptiveStepSizer, family="altar.bayesian.stepsizers.targetedrate"):
    """
    Multiplicative feedback that steers acceptance towards a goal using
    the standard exponential rule: step *= exp(gain * (ratio - target)).
    """

    target = altar.properties.float(default=0.7)
    target.doc = "desired acceptance probability"

    gain = altar.properties.float(default=1.0)
    gain.doc = "feedback gain controlling responsiveness"

    def recompute_step_size(self):
        # compute acceptance ratio
        ratio = (self._accepted_since_adjust / self._attempts_since_adjust
                 if self._attempts_since_adjust > 0 else 0.0)
        # return updated step size
        return self.step_size * math.exp(self.gain * (ratio - self.target))


class DualAveragingStepSize(AdaptiveStepSizer, family="altar.bayesian.stepsizers.dual"):
    """
    Dual-averaging step-size adaptation similar to NUTS:
    maintains an averaged log-step that slowly forgets past errors.
    """

    target = altar.properties.float(default=0.65)
    target.doc = "target acceptance probability"

    gamma = altar.properties.float(default=0.05)
    gamma.doc = "controls the speed of the dual averaging updates"

    t0 = altar.properties.float(default=10.0)
    t0.doc = "stabilizes the initial iterations"

    kappa = altar.properties.float(default=0.75)
    kappa.doc = "controls how quickly the running average forgets old iterations"

    @altar.export
    def initialize(self, value):
        initial = super().initialize(max(value, 1e-8))
        self._iteration = 0
        self._log_step = math.log(initial)
        self._log_avg = self._log_step
        self._mu = math.log(10 * initial)
        return initial

    @altar.export
    def adjust(self, *, value, ratio, attempts, accepted):
        self._iteration += 1
        h = self.target - ratio
        log_step = self._mu - (math.sqrt(self._iteration) / self.gamma) * h
        eta = self._iteration ** (-self.kappa)
        self._log_avg = eta * log_step + (1.0 - eta) * self._log_avg
        self._log_step = log_step
        return self._clip(math.exp(log_step))


# backward-compatible aliases
LinearAcceptanceStepSize = LinearRate
TargetAcceptanceStepSize = TargetedRate

# end of file
