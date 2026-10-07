#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Methods for Strike Slip faults

# externals
import numpy


def StressKernelStrike(x, z, xc, zc, width, mu):
    """
    Compute Stress K_{xy}(x,z) due to a slab with strike slip centered at (xc,zc) with width {width}
    Shear modulus {mu} default to 1 (as unit)
    """
    # get Pi
    from math import pi

    # upper edge z coordinate zc < 0
    z0 = zc+width/2
    # lower edge z coordinate
    z1 = zc-width/2
    # x-distance between source and obs, square
    xd2 = (x-xc)**2

    # shallow edge
    zd = z-z0
    Kxy = zd/(zd**2+xd2)
    # deeper edge
    zd = z-z1
    Kxy -= zd/(zd**2+xd2)
    # shallow edge image
    zd = z+z0
    Kxy -= zd/(zd**2+xd2)
    # deeper edge image
    zd = z+z1
    Kxy += zd/(zd**2+xd2)
    # factor
    Kxy *= mu/(2*pi)
    return Kxy


def DisplacementStrike(x, z, xc, zc, width):
    """
    Compute Displacement Uy(x,z) due to a slab with strike slip centered at (xc,zc) with width {width}
    """
    # get Pi
    from math import pi, atan2

    # upper edge z coordinate zc < 0
    z0 = zc+width/2
    # lower edge z coordinate
    z1 = zc-width/2
    # x-distance between source and obs, square
    xd = x-xc

    # displacement Uy from upper/lower edges and their images
    Uy = atan2(xd, z-z0) - atan2(xd, z-z1) - atan2(xd, z+z0) + atan2(xd, z+z1)
    return Uy/(2*pi)


def GenerateStressKernel(z0, z1, patches, mu=1):
    """
    Generate Stress Kernel for a strike fault between {z0, z1} into {patches}
    """
    width = (z0-z1)/patches
    halfwidth = width/2

    K = numpy.zeros(shape=(patches, patches))
    for i in range(patches):
        for j in range(i, patches):
            K[i,j] = StressKernelStrike(x=0, z=z0-i*width-halfwidth, xc=0, zc=z0-j*width-halfwidth, width=width, mu=mu)
            K[j,i] = K[i,j]
    return K


def GenerateDisplacementKernel(z0, z1, patches, stations):
    """
    Generate Displacement Kernel between a strike fault and {stations - x coordinate only}
    """
    # get the fault patch width
    width = (z0-z1)/patches
    halfwidth = width/2

    # get the number of stations
    nobs = stations.size

    # create the displacement kernel
    kernel = numpy.zeros(shape=(nobs, patches))
    for obs in range(nobs):
        xobs = stations[obs]
        zobs = 0
        for patch in range(patches):
            xsrc = 0
            zsrc = -patch*width - halfwidth
            kernel[obs,patch] = DisplacementStrike(x=xobs, z=zobs, xc=xsrc, zc=zsrc, width=width)
    return kernel


# end of file



