# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

#! /usr/bin/env python3

import numpy
import h5py

def PseudoInverse(gf_file, data_file):
    """
    Use pseudo inverse to solve linear inversion problem
    """

    #load green's function
    gf = numpy.loadtxt(gf_file)
    # load data
    d = numpy.loadtxt(data_file)

    # pinv
    gfinv = numpy.linalg.pinv(gf)

    # inversion
    theta = numpy.dot(gfinv, d)

    # print out the result
    print("The parameters given by the pseudoinverse matrix\n", theta)
    # if needed, uncomment the following line to save it as well
    #numpy.savetxt("theta_by_pinverse.txt", theta)


    h5 = h5py.File("results/step_final.h5", "r")
    thetab = numpy.array(h5['ParameterSets/pset'])
    theta_m = numpy.mean(thetab, axis=0)
    theta_std = numpy.std(thetab, axis=0)
    print("The parameters given by the Bayesian inversion (mean values)\n", theta_m)
    print("The difference between them are\n", theta-theta_m)
    print("which are expected to be smaller than the standard deviations from Bayesian simultion\n", theta_std)
    # all done
    return theta


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Use pseudoInverse matrix for inversion')
    parser.add_argument('--green', help="green's function file name", default="input/green.txt")
    parser.add_argument('--data', help="data file name", default="input/data.txt")
    args = parser.parse_args()

    theta = PseudoInverse(args.green, args.data)
