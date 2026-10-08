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
import typing
import numpy
# the package
import altar
# my protocol
from .RNG import RNG as rng


# the numpy bit generators, by name
generators: dict[str, type[numpy.random.BitGenerator]] = {
    "pcg64": numpy.random.PCG64,
    "pcg64dxsm": numpy.random.PCG64DXSM,
    "mt19937": numpy.random.MT19937,
    "philox": numpy.random.Philox,
    "sfc64": numpy.random.SFC64,
}


# declaration
class NumpyRNG(altar.component, family="altar.simulations.rng.numpy", implements=rng):
    """
    A numpy random number generator
    """

    # user configurable state
    seed = altar.properties.int(default=0)
    seed.doc = 'the number with which to seed the generator'

    algorithm = altar.properties.str(default='pcg64')
    algorithm.doc = 'the numpy bit generator: pcg64, pcg64dxsm, mt19937, philox or sfc64'
    algorithm.validators = altar.constraints.isMember(*generators)

    # public data
    rng: numpy.random.Generator


    # required behavior
    @altar.export
    def initialize(self, **kwds) -> typing.Self:
        """
        Initialize the random number generator
        """
        # nothing to do
        return self


    def reseed(self, rank: int) -> typing.Self:
        """
        Give the process of {rank} a stream of its own, derived from my {seed}; in place, so
        whoever holds my {rng} already draws from the new stream
        """
        fresh = self.generator(numpy.random.SeedSequence([self.seed, rank]))
        self.rng.bit_generator.state = fresh.bit_generator.state
        return self


    # meta-methods
    def __init__(self, **kwds) -> None:
        # chain  up
        super().__init__(**kwds)
        # build the random number generator
        self.rng = self.generator(numpy.random.SeedSequence(self.seed))
        # all done
        return


    # implementation details
    def generator(self, seed: numpy.random.SeedSequence) -> numpy.random.Generator:
        """
        A generator of my {algorithm} seeded from the {numpy.random.SeedSequence} {seed}
        """
        return numpy.random.Generator(generators[self.algorithm](seed))


    def show(self) -> typing.Self:
        """
        Display some information about me
        """
        # get the journal
        import journal
        # make a channel
        channel = journal.debug("altar.init")
        # show me
        channel.line(f"{self.pyre_name}:")
        channel.line(f"       seed: {self.seed}")
        channel.line(f"  algorithm: {self.algorithm}")
        channel.log()
        # all done
        return self


# end of file
