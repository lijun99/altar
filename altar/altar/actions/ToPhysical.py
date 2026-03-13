# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
# Author(s): Lijun Zhu
#

# get the package
import altar
import altar.cuda
# get my model package
import altar.models.seismic
# external packages
import h5py
import numpy as np


# declaration
class ToPhysical(altar.panel(), family='altar.actions.tophysical'):
    """
    Convert parameters from sampling to physical
    """

    h5file = altar.properties.path(default=None)
    h5file.doc = "h5 file to be converted"

    # commands
    @altar.export(tip="convert parameters from sampling to physical")
    def default(self, plexus, **kwds):
        """
        Convert parameters from sampling to physical and vice versa
        """
        # check the parameters
        if self.h5file is None:
            print("Usage: slipmodel.plexus tophysical --config=static.pfg --h5file=step_final.h5")
        else:
            # get the model and prior info
            model = plexus.model
            psets = model.psets
            psets_list = model.psets_list
            model_dtype = model.precision
            # enforce the job task to be one gpu
            plexus.job.tasks = 1
            plexus.job.gpus = 1

            # open the hdf5 file for conversion
            with h5py.File(self.h5file.path, 'r+') as f:
                # read theta from h5file
                psetsgrp = f['ParameterSets']
                if 'theta' in psetsgrp:
                    theta = psetsgrp['theta'][...]
                else:
                    # Determine the total number of columns for theta
                    total_columns = sum(dset.shape[1] for dset in psetsgrp.values())
                    # Initialize an empty list to store slices
                    slices = []
                    # Iterate over all parameter sets
                    for name in psets_list:
                        dset = psetsgrp[name][...]
                        slices.append(dset)
                    # Concatenate all slices along the second axis
                    theta = np.concatenate(slices, axis=1)

                # get data type (might be different from model_dtype)
                file_dtype = theta.dtype
                samples, parameters = theta.shape

                # convert theta to model_dtype
                theta = theta.astype(model_dtype)
                # read the Bayesian probabilities
                prior = f['Bayesian']['prior'][...].astype(model_dtype)
                likelihood = f['Bayesian']['likelihood'][...].astype(model_dtype)
                posterior = f['Bayesian']['posterior'][...].astype(model_dtype)
                print("mean values of prior, likelihood, posterior for sampling parameters: ")
                print(prior.mean(), likelihood.mean(), posterior.mean())

                # convert theta in gpu (since currently only defined in gpu)
                gtheta = altar.cuda.matrix(source=theta)
                # prior will be recomputed, only initialize
                gprior = altar.cuda.vector(shape=samples, dtype=model_dtype).zero()
                # posterior will be recomputed as likelihood+prior, so load likelihood at first
                gposterior = altar.cuda.vector(source=likelihood)

                # convert from sampling to physical
                print("converting from sampling to physical ...")
                for pset in psets.values():
                    if pset.prior.has_reparametrization:
                        pset.prior.cu_to_physical(theta=gtheta, batch=samples)
                print("recomputing prior and posterior for physical parameters ...")
                for pset in psets.values():
                    pset.prior.cu_eval_prior_physical(theta=gtheta, prior=gprior, batch=samples)
                gposterior += gprior

                # copy back to cpu
                gtheta.copy_to_host(target=theta)
                gprior.copy_to_host(target=prior)
                gposterior.copy_to_host(target=posterior)

                # print(theta.mean(axis=0), theta.std(axis=0))
                print("mean values of prior, likelihood, posterior for physical parameters: ")
                print(prior.mean(), likelihood.mean(), posterior.mean())

                # write back to h5file
                print("updating h5file with physical theta, prior and posterior ...")
                if 'theta' in psetsgrp:
                    psetsgrp['theta'][...] = theta.astype(file_dtype)
                else:
                    for name, pset in psets.items():
                        psetsgrp[name][...] = theta[:, pset.offset:pset.offset + pset.count].astype(file_dtype)
                f['Bayesian']['prior'][...] = prior.astype(file_dtype)
                f['Bayesian']['posterior'][...] = posterior.astype(file_dtype)
                print("all done!")

        # all done
        return


# end of file
