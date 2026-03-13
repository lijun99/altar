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


#
class Notifier(altar.component,
               family="altar.simulations.dispatchers.notifier",
               implements=altar.simulations.dispatcher):

    """
    A dispatcher of events generated during the annealing process
    """


    # constants: the event loop identifiers
    start = "simulation_start"

    sample_posterior_start = "sample_posterior_start"
    prepare_sampling_pdf_start = "prepare_sampling_pdf_start"
    prepare_sampling_pdf_finish = "prepare_sampling_pdf_finish"
    beta_start = "beta_start"
    walk_chains_start = "walk_chains_start"
    chain_advance_start = "chain_advance_start"
    verify_start = "verify_start"
    verify_finish = "verify_finish"
    prior_start = "prior_start"
    prior_finish = "prior_finish"
    data_start = "data_start"
    data_finish = "data_finish"
    posterior_start = "posterior_start"
    posterior_finish = "posterior_finish"
    accept_start = "accept_start"
    accept_finish = "accept_finish"
    chain_advance_finish = "chain_advance_finish"
    walk_chains_finish = "walk_chains_finish"
    resample_start = "resample_start"
    resample_finish = "resample_finish"
    beta_finish = "beta_finish"
    sample_posterior_finish = "sample_posterior_finish"

    finish = "simulation_finish"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize me given an {application} context
        """
        # nothing to do
        return self


    # interface
    def register(self, monitor):
        """
        Enable {monitor} as an observer of simulation events
        """
        # go through all known event types
        for event in self.events.keys():
            # check
            try:
                # whether the {monitor} implements a handler for this particular event
                handler = getattr(monitor, event)
            # if not
            except AttributeError:
                # no worries
                continue
            # otherwise, add the handler to the correct event table entry
            self.events[event].addObserver(handler)
        # all done
        return


    def notify(self, event, controller):
        """
        Notify all handlers that are waiting for {event}
        """
        # find the observable associated with this event
        observable = self.events[event]
        # and ask it to notify its observers
        observable.notifyObservers(controller=controller)
        # all done
        return


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)

        # establish the table of handled events
        self.events = {
            self.start: altar.patterns.observable(),
            self.sample_posterior_start: altar.patterns.observable(),
            self.prepare_sampling_pdf_start: altar.patterns.observable(),
            self.prepare_sampling_pdf_finish: altar.patterns.observable(),
            self.beta_start: altar.patterns.observable(),
            self.walk_chains_start: altar.patterns.observable(),
            self.chain_advance_start: altar.patterns.observable(),
            self.verify_start: altar.patterns.observable(),
            self.verify_finish: altar.patterns.observable(),
            self.prior_start: altar.patterns.observable(),
            self.prior_finish: altar.patterns.observable(),
            self.data_start: altar.patterns.observable(),
            self.data_finish: altar.patterns.observable(),
            self.posterior_start: altar.patterns.observable(),
            self.posterior_finish: altar.patterns.observable(),
            self.accept_start: altar.patterns.observable(),
            self.accept_finish: altar.patterns.observable(),
            self.chain_advance_finish: altar.patterns.observable(),
            self.walk_chains_finish: altar.patterns.observable(),
            self.resample_start: altar.patterns.observable(),
            self.resample_finish: altar.patterns.observable(),
            self.sample_posterior_finish: altar.patterns.observable(),
            self.beta_finish: altar.patterns.observable(),
            self.finish: altar.patterns.observable(),
        }

        # all done
        return


# end of file
