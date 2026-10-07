#!/usr/bin/env python3

# -*- python -*-
# -*- coding: utf-8 -*-
#
# eric m. gurrola <eric.m.gurrola@jpl.nasa.gov>
#
# (c) 2018-2021 jet propulsion laboratory
# (c) 2018-2021 california institute of technology
# (c) 2018-2021 parasim
# all rights reserved
#

"""
Test for CDM: Compound Dislocation Model
"""

import sys
import numpy
from altar.models.cdm.libcdm import CDM

# Matlab test coordinates and displacements in EFCS (East, North, Vertical).
# Sampled from computed output from original Matlab code.
# inputs

# matlab input X coordinates sample
X = numpy.array([-7,    -6.90, -6.88, -6.86,
                 -6.84, -6.82,  0,     0.50,
                  1.10,  6.50,  6.86,  7])
# matlab input Y coordinates sample
Y = numpy.array([-5,    -5,    -5,    -5,
                 -5,    -5,    -4.92, -4.92,
                 -4.92, -4.92,  4.94, 5])

# matlab de corresponding outputs
mde = numpy.array([-4.831270e-06, -4.886526e-06, -4.897588e-06, -4.908654e-06,
                  -4.919721e-06, -4.930791e-06, -1.331881e-06,  9.160793e-08,
                   1.801943e-06,  5.766744e-06,  4.544161e-06,  4.439410e-06])
# matlab dn corresponding outputs
mdn = numpy.array([-2.988161e-06, -3.064582e-06, -3.080126e-06, -3.095757e-06,
                  -3.111477e-06, -3.127285e-06, -1.312579e-05, -1.331642e-05,
                  -1.315063e-05, -4.467011e-06,  3.648027e-06,  3.525858e-06])
# matlab dv corresponding outputs
mdv = numpy.array([ 1.799845e-06,  1.844885e-06,  1.854042e-06,  1.863249e-06,
                    1.872507e-06,  1.881815e-06,  7.624726e-06,  7.706569e-06,
                    7.578237e-06,  2.489978e-06,  1.996341e-06,  1.909694e-06])

def compare(name, de, dn, dv, tolerance=1e-6):
    """
    Compare the displacements against the matlab reference; return the number of mismatches
    """
    failures = 0
    for m, u, label in ((mde, de, "de"), (mdn, dn, "dn"), (mdv, dv, "dv")):
        error = numpy.abs((numpy.asarray(u) - m) / m)
        if error.max() > tolerance:
            print(f"{name}: {label} differs from matlab by up to {error.max():.2e}")
            failures += 1
    return failures


def main(X0, Y0, depth, omegaX, omegaY, omegaZ, ax, ay, az, opening, nu, verbose=False):
    """
    Test the python and c++ implementations of CDM against the matlab output
    X0, Y0, depth: define the position of the dislocation
    omegaX, omegaY, omegaZ: define the orientation (clockwise rotations) of the dislocation
    ax, ay, az: define the semi-axes  of the dislocation in the "body fixed" coordinates
    opening: the tensile component of the Burgers vector
    """
    # the python implementation
    de, dn, dv = CDM(X=X, Y=Y, X0=X0, Y0=Y0, depth=depth,
                     omegaX=omegaX, omegaY=omegaY, omegaZ=omegaZ,
                     ax=ax, ay=ay, az=az, opening=opening, nu=nu)
    failures = compare("libcdm.py", de, dn, dv)

    # the c++ implementation, if it was built
    try:
        from altar.models.cdm.ext import libcdm
    except ImportError:
        libcdm = None
    if libcdm is not None:
        u = numpy.array(libcdm.enu(
            (X0, Y0, depth, opening, ax, ay, az, omegaX, omegaY, omegaZ), list(X), list(Y), nu))
        failures += compare("libcdm", u[:, 0], u[:, 1], u[:, 2])
    elif verbose:
        print("the c++ extension is not available")

    return 1 if failures else 0


if __name__ == "__main__":

    verbose = False
    if len(sys.argv) > 1:
        verbose = True

    # Set test parameter values

    # Horizontal coordinates (in EFCS) and depth of the point CDM. The depth must be a positive
    # value. X0, Y0, and depth have the same units as X, Y, Z.
    X0, Y0, depth = 0.5, -0.25, 2.75

    # Rotation angles (clockwise) in degrees defining the orientation of the point CDM.
    omegaX, omegaY, omegaZ = 5., -8., 30.

    # Semi-axes of the CDM along the X, Y, Z axes before applying the rotation
    #  (ax, ay, az have the same units as X and Y.)
    ax, ay, az = 0.4, 0.45, 0.8

    # opening: the tensile component of the Burgers vector of the rectangular dislocation that
    # form the CDM. The unit of opening must be the same as the unit of ax, ay, az
    opening = 0.001

    # Poisson's ratio
    nu = 0.25

    # run the test
    status = main(X0, Y0, depth, omegaX, omegaY, omegaZ, ax, ay, az, opening, nu,
                  verbose=verbose)

    # communicate status
    raise SystemExit(status)

# end-of-file
