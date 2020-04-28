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
class Linear(BayesianL2, family="altar.models.regression.linear"):
    """
    Linear Regression model y= ax +b
    """

    # additional model properties
    x_file = altar.properties.path(default='x.txt')
    x_file.doc = "the input file for x variable"

    @altar.export
    def initialize(self, application):
        """
        Initialize the state of the model
        """
        # model specific initialization before superclass
        # none for this model

        # call the super class initialization
        super().initialize(application=application)

        # model specific initialization after superclass
        # grab data
        self.x = self.loadFile(self.x_file)
        self.y = self.dataobs.dataobs
        # set the return_residual flag
        # forward model calculates the residual between prediction and data
        self.return_residual = True
        # all done, return self
        return self


    def forwardModel(self, theta, prediction):
        """
        Forward Model
        :param theta: sampling parameters for one sample
        :param prediction: data prediction or residual (prediction - observation)
        :return: none
        """

        # grab the parameters from theta

        slope = theta[0]
        intercept = theta[1]

        # calculate the residual between data prediction and observation
        size = self.observations
        for i in range(size):
            prediction[i] = slope * self.x[i] + intercept - self.y[i]

        # all done
        return self


    # private variables
    x = None
    y = None

# end of file
