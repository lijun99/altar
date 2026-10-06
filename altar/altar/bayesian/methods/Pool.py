# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# declaration
class Pool:
    """
    The states the chains keep while they walk: every {interval} steps, into the next of {size}
    slots of {chains} rows of the population, round robin, and their final state always; slots a
    short walk leaves unfilled get the final state too
    """


    # interface
    def begin(self):
        """
        Start the bookkeeping of a walk
        """
        self.count = 0
        self.kept = 0
        self.last = None
        return self


    def advance(self, keep):
        """
        A step of the walk is done; {keep(offset)} copies the chains into the population rows
        from {offset} on, if this one is to be kept
        """
        self.count += 1
        if self.size > 1 and self.count % self.interval == 0:
            self._keep(keep)
        return self


    def end(self, keep):
        """
        The walk is done: keep its final state, and fill the slots it left unfilled
        """
        if self.last != self.count:
            self._keep(keep)
        while self.kept < self.size:
            self._keep(keep)
        return self


    # meta-methods
    def __init__(self, size, interval, chains, **kwds):
        super().__init__(**kwds)
        self.size = size
        self.interval = interval
        self.chains = chains
        self.slot = 0
        self.begin()
        return


    # implementation details
    def _keep(self, keep):
        keep(self.slot * self.chains)
        self.slot = (self.slot + 1) % self.size
        self.kept += 1
        self.last = self.count
        return


# end of file
