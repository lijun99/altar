# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# the package
import altar
from altar.cuda import libcudaaltar
from altar.cuda import cublas
# my protocol
from altar.norms.L2 import L2


# declaration
class cudaL2(L2, family="altar.norms.cudal2"):
    """
    The L2 norm
    """

    # interface
    @altar.export
    def cuEval(self, data, out=None, batch=None, cdinv=None):
        """
        Compute the L2 norm of the given data  ||x||
        Arguments:
            data - matrix (samples x observations) 
            batch - number of samples to be computed (first rows)
            cdinv - inverse covariance matrix (observations x observations) in its Cholesky decomposed form (Upper Triangle)   
        Return:
            out - norm vector (samples)  
        """

        samples = data.shape[0]
        # no batch provided, compute all samples
        if batch is None or batch > samples:
            batch = samples

        # if output is not pre-allocated, allocate it
        if out is None:
            out = altar.cuda.vector(shape=samples)

        # if covariance matrix is provided
        if cdinv is not None:
            # data [samples][obs] = data[samples][obs] x cdinv[obs][obs]
            cublas.trmm(cdinv, data, out=data,
                alpha=1.0, uplo=cublas.FillModeUpper, side=cublas.SideRight,
                transa = cublas.OpNoTrans, diag=cublas.DiagNonUnit)
                
        # compute the norm 
        libcudaaltar.cudaL2_norm(data.data, out.data, batch)
        # return the result
        return out


    def cuEvalLikelihood(self, data, constant=0.0, out=None, batch=None, cdinv=None):
        """
        Compute the L2 norm data likelihood of the given data  const - ||x||^2/2
        Arguments:
            data - matrix (samples x observations) 
            batch - number of samples to be computed (first rows)
            constant - normalization constant
            cdinv - inverse covariance matrix (observations x observations) in its Cholesky decomposed form (Upper Triangle)   
        Return:
            out -  data likelihood vector (samples)  
        """

        samples = data.shape[0]
        # no batch provided, compute all samples
        if batch is None or batch > samples:
            batch = samples

        # if output is not pre-allocated, allocate it
        if out is None:
            out = altar.cuda.vector(shape=samples)

        # if covariance matrix is provided
        if cdinv is not None:
            # data [samples][obs] = data[samples][obs] x cdinv[obs][obs]
            cublas.trmm(cdinv, data, out=data,
                alpha=1.0, uplo=cublas.FillModeUpper, side=cublas.SideRight,
                transa = cublas.OpNoTrans, diag=cublas.DiagNonUnit)
                
        # compute the norm 
        libcudaaltar.cudaL2_normllk(data.data, out.data, batch, constant)
        # return the result
        return out

# end of file
