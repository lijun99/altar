# -*- python -*-
# -*- coding: utf-8 -*-
#
# A scheduler that keeps beta=1 (temperature=1) always
#

import altar

class ConstantTemperature(altar.component, family="altar.schedulers.constant"):
    """
    A scheduler that keeps the temperature fixed at 1 (beta=1)
    """

    @altar.provides
    def initialize(self, application):
        """
        Initialize the scheduler (no-op for constant temperature)
        """
        return self

    @altar.provides
    def update(self, step):
        """
        No update needed; just return the step
        """
        return step

    @altar.provides
    def update_temperature(self, step):
        """
        Set beta to 1 (temperature=1)
        """
        step.beta = 1.0
        return step

    @altar.provides
    def compute_covariance(self, step):
        """
        Optionally compute covariance (no-op for constant temperature)
        """
        return step

    @altar.provides
    def rank(self, step):
        """
        Optionally rank samples (no-op for constant temperature)
        """
        return step

# end of file