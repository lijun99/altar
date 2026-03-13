# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#


# the package
import altar
# my protocol
from .Monitor import Monitor as monitor


# an implementation of the monitor protocol
class Reporter(altar.component, family="altar.simulations.monitors.reporter", implements=monitor):
    """
    Reporter reports simulation progress by using application journal channels
    """


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize me given an {application} context
        """
        # nothing to do
        return self


    # implementation details
    def simulation_start(self, controller, **kwds):
        """
        Handler invoked when the simulation is about to start
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: start")
        # all done
        return


    def sample_posterior_start(self, controller, **kwds):
        """
        Handler invoked at the beginning of sampling the posterior
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: sample_posterior_start")
        # all done
        return


    def prepare_sampling_pdf_start(self, controller, **kwds):
        """
        Handler invoked at the beginning of the preparation of the sampling PDF
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: prepare_sampling_pdf_start")
        # all done
        return


    def prepare_sampling_pdf_finish(self, controller, **kwds):
        """
        Handler invoked at the end of the preparation of the sampling PDF
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: prepare_sampling_pdf_finish")
        # all done
        return


    def beta_start(self, controller, **kwds):
        """
        Handler invoked at the beginning of the beta step
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: beta_start")
        # all done
        return


    def walk_chains_start(self, controller, **kwds):
        """
        Handler invoked at the beginning of the chain walk
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: walk_chains_start")
        # all done
        return


    def chain_advance_start(self, controller, **kwds):
        """
        Handler invoked at the beginning of a single step of chain walking
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: chain_advance_start")
        # all done
        return


    def chain_advance_finish(self, controller, **kwds):
        """
        Handler invoked at the end of a single step of chain walking
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: chain_advance_finish")
        # all done
        return


    def verify_start(self, controller, **kwds):
        """
        Handler invoked before we start verifying the generated sample
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: verify_start")
        # all done
        return


    def verify_finish(self, controller, **kwds):
        """
        Handler invoked after we are done verifying the generated sample
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: verify_finish")
        # all done
        return


    def prior_start(self, controller, **kwds):
        """
        Handler invoked before we compute the prior
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: prior_start")
        # all done
        return


    def prior_finish(self, controller, **kwds):
        """
        Handler invoked after we compute the prior
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: prior_finish")
        # all done
        return


    def data_start(self, controller, **kwds):
        """
        Handler invoked before we compute the data likelihood
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: data_start")
        # all done
        return


    def data_finish(self, controller, **kwds):
        """
        Handler invoked after we compute the data likelihood
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: data_finish")
        # all done
        return


    def posterior_start(self, controller, **kwds):
        """
        Handler invoked before we assemble the posterior
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: posterior_start")
        # all done
        return


    def posterior_finish(self, controller, **kwds):
        """
        Handler invoked after we assemble the posterior
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: posterior_finish")
        # all done
        return


    def accept_start(self, controller, **kwds):
        """
        Handler invoked at the beginning of sample accept/reject
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: accept_start")
        # all done
        return


    def accept_finish(self, controller, **kwds):
        """
        Handler invoked at the end of sample accept/reject
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: accept_finish")
        # all done
        return


    def walk_chains_finish(self, controller, **kwds):
        """
        Handler invoked at the end of the chain walk
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: walk_chains_finish")
        # all done
        return


    def resample_start(self, controller, **kwds):
        """
        Handler invoked before we start resampling
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: resample_start")
        # all done
        return


    def resample_finish(self, controller, **kwds):
        """
        Handler invoked after we are done resampling
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: resample_finish")
        # all done
        return


    def beta_finish(self, controller, **kwds):
        """
        Handler invoked at the end of the beta step
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: beta_finish")
        # all done
        return


    def sample_posterior_finish(self, controller, **kwds):
        """
        Handler invoked at the end of sampling the posterior
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: sample_posterior_finish")
        # all done
        return


    def simulation_finish(self, controller, **kwds):
        """
        Handler invoked when the simulation is about to finish
        """
        # grab a channel
        channel = controller.info
        # say something
        channel.log(f"{self.pyre_name}: finish")
        # all done
        return


# end of file
