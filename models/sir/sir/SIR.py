# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# author(s): Lijun Zhu


# the package
import altar
from altar.models.BayesianL2 import BayesianL2

# declaration
class SIR(BayesianL2, family="altar.models.sir"):
    """
    SIR model in epidemiology
    This model assumes five parameters
      S0 - The initial value of susceptible population (times the population base factor)
      I0 - The initial value of infectious
      R0 - The initial value of Recovered/Deaths
      β - the average number of contacts per person per time
      γ -  the rate of recovery or mortality
    to generate the time sequence of S(t), I(t) and R(t)
    It uses the new cases per day as data observations, or S(t-1)-S(t)
    """

    # configurable properties
    population_base = altar.properties.float(default=10000)
    population_base.doc = 'the base factor for population'


    def _SIR_Rate(self, S, I, N, β, γ):
        """
        SIR Rate equation
        """
        new = β*I*S/N
        recovered = γ*I
        Sn = S - new
        In = I + new - recovered
        return Sn, In

    def forwardModel(self, theta, prediction):
        """
        Forward SIR model
        """
        # grab the parameters from theta
        S0 = theta[0]*self.population_base
        I0 = theta[1]
        R0 = theta[2]
        # total number of people, a constant
        N = S0 + I0 + R0
        β = theta[3]
        γ = theta[4]

        # grab the observations
        obs = self.dataobs.dataobs
        days = self.observations

        # rate equation iterative solver
        for day in range(days):
            # get the S, I for a new day
            S, I = self._SIR_Rate(S0, I0, N, β, γ)

            # get the daily new cases
            prediction[day] = (S0 - S)-obs[day]

            # assign the new values of S, I for new day
            S0 = S
            I0 = I

        # all done
        return self

# end of file
