# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
from __future__ import annotations
import typing
import numpy

if typing.TYPE_CHECKING:
    from altar.simulations.Archiver import Archiver


# declaration
class Diagnostics:
    """
    How well the walks of an annealing run mix, β step by β step, read off the population of
    the manager: the parents a resampling keeps; the lineages, the initial samples the
    population descends from; the correlation, across the chains, of the final state of each
    chain with the one it started from; and, with a pool, the effective states a chain keeps,
    the variance of the population over that of the means of the chains

    The population holds the blocks of the {tasks} in order, each {pool} slots of its {chains};
    the chains start their walk from the first slot
    """


    # interface
    def resampled(self, rows: numpy.ndarray) -> typing.Self:
        """
        The population was replaced by its rows {rows}
        """
        self.parents = numpy.unique(rows).size
        self.lineage = self.lineage[rows]
        return self


    def begin(self, θ: numpy.ndarray) -> typing.Self:
        """
        The chains are about to walk from the population {θ}
        """
        self.start = numpy.array(θ[self.rows(slot=0)])
        return self


    def end(self, θ: numpy.ndarray, final: int, iteration: int, beta: float) -> dict:
        """
        The chains walked to the population {θ}, with their final states in slot {final}
        """
        θ = numpy.asarray(θ)
        self.correlation = correlation(self.start, θ[self.rows(slot=final)])
        self.states = None
        if self.pool > 1:
            # the slots of each chain, over the tasks, as (tasks, pool, chains, parameters)
            kept = θ.reshape(self.tasks, self.pool, self.chains, -1)
            means = kept.mean(axis=1, dtype=numpy.float64).reshape(-1, θ.shape[1])
            between = means.var(axis=0)
            self.states = θ.var(axis=0, dtype=numpy.float64) / numpy.where(between > 0, between, numpy.nan)
        # every state a chain keeps descends from the row it started from
        lineage = self.lineage[self.rows(slot=0)].reshape(self.tasks, 1, self.chains)
        self.lineage = numpy.broadcast_to(lineage, (self.tasks, self.pool, self.chains)).ravel()
        record = {
            "iteration": iteration,
            "beta": beta,
            "parents": self.parents,
            "lineages": numpy.unique(self.lineage).size,
            "correlation": numpy.nanmedian(self.correlation),
            "correlation_max": numpy.nanmax(numpy.abs(self.correlation)),
            "states": numpy.nan if self.states is None else numpy.nanmedian(self.states),
        }
        self.history.append(record)
        self.start = None
        self.parents = self.lineage.size
        return record


    def record(self, archiver: Archiver) -> typing.Self:
        """
        Write my history, and the correlation and effective states of the last walk, to {archiver}
        """
        if not self.history:
            return self
        for key in self.history[0]:
            archiver.write(f"Diagnostics/{key}", numpy.asarray([row[key] for row in self.history]))
        archiver.write("Diagnostics/walk/correlation", self.correlation)
        if self.states is not None:
            archiver.write("Diagnostics/walk/states", self.states)
        return self


    def rows(self, slot: int) -> numpy.ndarray:
        """
        The rows of the population that hold the chains in {slot}, task by task
        """
        block = self.chains * self.pool
        tasks = numpy.arange(self.tasks)[:, None] * block
        return (tasks + slot * self.chains + numpy.arange(self.chains)).ravel()


    # meta-methods
    def __init__(self, tasks: int, chains: int, pool: int, **kwds) -> None:
        super().__init__(**kwds)
        self.tasks = tasks
        self.chains = chains
        self.pool = pool
        # every initial sample is a lineage of its own
        self.lineage = numpy.arange(tasks * pool * chains)
        self.parents = self.lineage.size
        self.history = []
        self.start = None
        self.correlation = None
        self.states = None
        return


def correlation(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    """
    The correlation of each column of {a} with the same column of {b}, across the rows; nan for
    a column that doesn't vary
    """
    a = a - a.mean(axis=0, dtype=numpy.float64)
    b = b - b.mean(axis=0, dtype=numpy.float64)
    scale = numpy.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return (a * b).sum(axis=0) / numpy.where(scale > 0, scale, numpy.nan)


# end of file
