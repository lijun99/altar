#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

import h5py
import numpy
import sys

def checkDataDiff():
    """
    Check the difference between data predictions and observations
    """

    # file names for data, modify them accordingly
    dataPrediction = "forward_prediction.h5"
    staticData = "9patch/static.data.h5"
    kinematicData = "9patch/kinematicG.data.h5"

    # open data prediction file and get predictions for one or both models
    h5file = h5py.File(dataPrediction, 'r')
    staticDataPrediction = numpy.asarray(h5file.get("static.Data"))
    kinematicDataPrediction = numpy.asarray(h5file.get("kinematic.Data"))
    h5file.close()

    # get data observation for static model
    h5file = h5py.File(staticData, 'r')
    staticDataObservation = numpy.asarray(h5file.get("static.data"))
    h5file.close()

    # get data observation for kinematic model
    h5file = h5py.File(kinematicData, 'r')
    kinematicDataObservation = numpy.asarray(h5file.get("kinematicG.data"))
    h5file.close()


    # check difference

    # max error and relative error
    error_max = 1.e-3
    error_rel_max = 1.e-1

    print("checking static model ...")
    # compute the relative difference
    diff = staticDataPrediction - staticDataObservation
    diff_ratio =diff/staticDataObservation
    diff_count = 0
    for i in range(diff.size):
        if abs(diff[i]) > error_max and abs(diff_ratio[i])> error_rel_max:
            print(f"Difference at {i}, with " +
                f"pred {staticDataPrediction[i]} " +
                f"obs {staticDataObservation[i]}")
            diff_count += 1
    print(f"There are {diff_count} data points out of {diff.size} with large differences")

    print("checking kinematic model ...")
    # compute the relative difference
    diff = kinematicDataPrediction - kinematicDataObservation
    diff_ratio = diff/kinematicDataObservation
    diff_count = 0
    for i in range(diff.size):
        if abs(diff[i]) > error_max and abs(diff_ratio[i])> error_rel_max:
            print(f"Difference at {i}, with " +
                  f"pred {kinematicDataPrediction[i]} " +
                  f"obs {kinematicDataObservation[i]}")
            diff_count += 1
    print(f"There are {diff_count} data points out of {diff.size} with large differences")

    # all done
    return


if __name__ == "__main__":
    checkDataDiff()

