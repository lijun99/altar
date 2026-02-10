# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# export my parts
from . import (
    # norms
    norms,
    # probability distribution functions
    distributions,
    # support for Bayesian explorations using Markov chain Monte Carlo
    bayesian,
    models,
    data,
    ext,
    )

from cuda import (
    vector,
    matrix,
    curand,
    cublas,
    Device,
    manager,
    stats,
    libcuda,
    use_device,
    )

# my extension modules
from .ext import cudaaltar as libcudaaltar


def get_current_device():
    """
    Return current cuda device
    """
    return manager.current_device

def curand_generator():
    device = get_current_device()

    return device.get_curand_generator()

def cublas_handle():
    device = get_current_device()
    return device.get_cublas_handle()

# administrative
def copyright():
    """
    Return the altar copyright note
    """
    return print(meta.header)


def license():
    """
    Print the altar license
    """
    # print it
    return print(meta.license)


def version():
    """
    Return the altar version
    """
    return meta.version


def credits():
    """
    Print the acknowledgments
    """
    # print it
    return print(meta.acknowledgments)
