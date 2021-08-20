#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author(s): Lijun Zhu

import h5py
import numpy


def MeanModel():
    """
    Compute the mean model of the AlTar step results
    Also convert the AlTar2 output to theta matrix format in AlTar-1.1
    """


    # open input/output
    input = h5py.File('step_final.h5', 'r')
    output = h5py.File('step_final_v1.h5', 'w')
    # use the psets_list as in the model pfg file
    psets_list = ['strikeslip', 'dipslip', 'risetime', 'rupturevelocity', 'hypocenter']

    # get the number of samples
    theta = numpy.asarray(input.get('ParameterSets/'+psets_list[0]))
    samples = theta.shape[0]

    # create an empty array
    theta=numpy.empty(shape=(samples,0), dtype=theta.dtype)
    for pset_name in psets_list:
        pset = numpy.array(input.get('ParameterSets/'+pset_name))
        theta=numpy.concatenate((theta, pset), axis=1)

    output.create_dataset('Sample Set', data=theta)
    print("converted sample set of size:", theta.shape)

    # convert bayesian prob/llk
    prior = numpy.asarray(input.get('Bayesian/prior'))
    output.create_dataset('Prior Log-likelihood', data=prior)
    data = numpy.asarray(input.get('Bayesian/likelihood'))
    output.create_dataset('Data Log-likelihood', data=data)
    posterior = numpy.asarray(input.get('Bayesian/posterior'))
    output.create_dataset('Posterior Log-likelihood', data=posterior)
    print("converted likelihood")

    # convert Covariance
    covariance = numpy.asarray(input.get('Annealer/covariance'))
    output.create_dataset('Covariance', data=covariance)
    beta = numpy.asarray(input.get('Annealer/beta'))
    output.create_dataset('Beta', data=beta)
    print("converted covariance matrix")

    output.close()
    input.close()

    # compute the mean model
    mean = theta.mean(axis=0)
    std = theta.std(axis=0)

    print("Mean model and std are saved to text files. Here are the first 10 parameters ... ")
    for i in range(min(10, mean.size)):
        print(f"{i}: ({mean[i]} +/- {std[i]})")

    numpy.savetxt("theta_mean.txt", mean)
    numpy.savetxt("theta_std.txt", std)


if __name__ == "__main__":
    MeanModel()



