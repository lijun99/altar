# -*- python -*-
# -*- coding: utf-8 -*-
#
# A scheduler that keeps beta fixed at a constant value (default 1)
#

import altar
from .Scheduler import Scheduler as scheduler

class ConstantTemperature(altar.component, family="altar.schedulers.constant", implements=scheduler):
    """
    A scheduler that keeps the temperature fixed at a constant beta value.
    """

    beta_start = altar.properties.float(default=1.0)
    beta_start.doc = "the fixed beta value used for the entire run"

    @altar.export
    def initialize(self, application):
        return self

    @altar.export
    def update(self, step):
        self.update_temperature(step=step)
        step.compute_posterior()
        return step

    @altar.export
    def update_temperature(self, step):
        step.beta = self.beta_start
        return step

    @altar.export
    def compute_covariance(self, step):
        return step

    @altar.export
    def rank(self, step):
        return step

# end of file
