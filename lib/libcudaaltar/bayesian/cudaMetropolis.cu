// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2019 parasim inc
// (c) 2010-2019 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

// declarations 
#include "cudaMetropolis.h"
// cuda utitlities
#include <pyre/cuda.h>
#include <iostream>

// cuda kernel declarations
namespace cudaMetropolis_kernels {
    
    __global__ void _setValidSampleIndices(int * const valid_sample_indices, const int * const invalid,
        const int samples, int * valid_samples
        );

    template <typename realtype_t>
    __global__ void _queueValidSamples(realtype_t * const theta_candidate, const realtype_t * const theta_proposal,
        const int * const validSample_indices, const size_t samples, const size_t parameters
        );
        
    template <typename realtype_t>
    __global__ void _metropolisUpdate(realtype_t * const theta, realtype_t * const prior, 
        realtype_t * const data, realtype_t * const posterior,  
        const realtype_t * const theta_candidate, const realtype_t * const prior_candidate,
        const realtype_t * const data_candidate, const realtype_t * const posterior_candidate,
        const realtype_t * const dices, int * const accpetance_flag, const int * const valid_sample_indices,
        const int samples, const int parameters
        );    
}  
 
// set valid sample indices for queueing and Metropolis update
//        invalid samples are not updated 
void altar::cuda::bayesian::cudaMetropolis::
setValidSampleIndices(int * const valid_sample_indices, const int * const invalid, 
    const int samples, int * valid_samples, 
    cudaStream_t stream) 
{
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(samples, blockSize);
    // allocate a device variable for valid sample counts
    cudaSafeCall(cudaMemset(valid_samples, 0, sizeof(int)));
    cudaSafeCall(cudaMemset(valid_sample_indices, 0, samples*sizeof(int)));
    cudaMetropolis_kernels::_setValidSampleIndices<<<gridSize, blockSize, 0, stream>>> (
        valid_sample_indices, invalid, samples, valid_samples);
    cudaCheckError("cudaMetropolis:setValidSampleIndices Error");  
}
 
 
/// @brief queue valid samples to a new theta
/// @param samples: batch or valid samples 
template <typename realtype_t>
void altar::cuda::bayesian::cudaMetropolis::
queueValidSamples(realtype_t * const theta_candidate, const realtype_t * const theta_proposal, 
    const int * const validSample_indices,
    const size_t samples, const size_t parameters, 
    cudaStream_t stream)
{
    // determine the block/grid size
    // one thread for one 
    dim3 blockSize (BDIMX, 1, BDIMY); // parameters, 1, batch/samples
    dim3 gridSize (IDIVUP(parameters, blockSize.x), 1, IDIVUP(samples, blockSize.z));

    // call cuda kernels
    cudaMetropolis_kernels::_queueValidSamples<realtype_t><<<gridSize, blockSize, 0, stream>>>(
        theta_candidate, theta_proposal, validSample_indices, samples, parameters);
    cudaCheckError("cudaMetropolis:queueValidSamples Error");
} 

// explicit instantiation
template void altar::cuda::bayesian::cudaMetropolis::queueValidSamples<float>(float * const, const float * const, const int * const, 
    const size_t, const size_t, cudaStream_t);
template void altar::cuda::bayesian::cudaMetropolis::queueValidSamples<double>(double * const, const double * const, const int * const, 
    const size_t, const size_t, cudaStream_t);


/// @brief Use Metropolis-Hastings algorithm to judge whether updates are accepted 
/// @param [in] samples: batch or number of valid samples from verification
template <typename realtype_t>
void altar::cuda::bayesian::cudaMetropolis::
metropolisUpdate(realtype_t * const theta, realtype_t * const prior, 
    realtype_t * const data, realtype_t * const posterior,  
    const realtype_t * const theta_candidate, const realtype_t * const prior_candidate, 
    const realtype_t * const data_candidate, const realtype_t * const posterior_candidate,
    const realtype_t * const dices, int * const acceptance_flag, const int * const valid_sample_indices,
    const int batch, const int parameters, 
    cudaStream_t stream)
{
    // determine the block/grid size
    // one thread for one sample
    int blockSize = NTHREADS;
    int gridSize = IDIVUP(batch, blockSize);
    cudaMetropolis_kernels::_metropolisUpdate<realtype_t><<<gridSize, blockSize, 0, stream>>>(
        theta, prior, data, posterior,
        theta_candidate, prior_candidate, data_candidate, posterior_candidate, 
        dices, acceptance_flag, valid_sample_indices,
        batch, parameters
        );
    cudaCheckError("cudaMetropolis:metropolisUpdate Error");
}
  
// explicit instantiation
template void altar::cuda::bayesian::cudaMetropolis::
    metropolisUpdate<float>(float * const, float * const, float * const, float * const,  
                            const float * const, const float * const, const float * const, const float * const,
                            const float * const, int * const, const int * const, const int, const int, 
                            cudaStream_t);
template void altar::cuda::bayesian::cudaMetropolis::
    metropolisUpdate<double>(double * const, double * const, double * const, double * const,  
                            const double * const, const double * const, const double * const, const double * const,
                            const double * const, int * const, const int * const, const int, const int, 
                            cudaStream_t);

// set valid sample indices  
__global__ void
cudaMetropolis_kernels::
_setValidSampleIndices(int * const valid_sample_indices, const int * const invalid, const int samples, int * valid_samples)
{
    //
    // get the thread id
    const int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= samples) return;

    int index_to_fill;
    // if it is a good sample
    if ( !(invalid[id]) )
    {
        // get the index_to_fill by an atomicAdd operation, which will increment the number of good samples
        // and return the previous number of good samples 
        index_to_fill = atomicAdd(valid_samples, 1);
        // save the index of this good sample by pushing to the end of goodsample_indices
        valid_sample_indices[index_to_fill] = id;
    }
}

//queue valid samples: move valid samples to top 
template <typename realtype_t>
__global__ void
cudaMetropolis_kernels::
_queueValidSamples(realtype_t * const theta_candidate, const realtype_t * const theta_proposal, 
    const int * const valid_sample_indices,
    const size_t samples, const size_t parameters)
{
    int sample = blockIdx.z*blockDim.z + threadIdx.z;
    int parameter = blockIdx.x*blockDim.x + threadIdx.x; 
    if (sample >= samples || parameter >= parameters) return;
     
    // IDX2R (row, col, ncols)
    theta_candidate[IDX2R(sample, parameter, parameters)] 
        = theta_proposal[IDX2R(valid_sample_indices[sample], parameter, parameters)];
}

// metropolis acceptance/rejection
template <typename realtype_t>
__global__ void
cudaMetropolis_kernels::
_metropolisUpdate(realtype_t * const theta, realtype_t * const prior, realtype_t * const data, realtype_t * const posterior,  
    const realtype_t * const theta_candidate, const realtype_t * const prior_candidate, 
    const realtype_t * const data_candidate, const realtype_t * const posterior_candidate,
    const realtype_t * const dices, int * const acceptance_flag, const int * const valid_sample_indices,
    const int samples, const int parameters)
{
    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= samples) return;
    
    int sample_index = valid_sample_indices[sample];
    if(log(dices[sample]) <= posterior_candidate[sample]-posterior[sample_index]) 
    {
        // acceptance
        // copy theta
        for(int parameter=0; parameter < parameters; parameter++)
            theta[IDX2R(sample_index, parameter, parameters)]  
                = theta_candidate[IDX2R(sample, parameter, parameters)];
        // copy densitities 
        prior[sample_index] = prior_candidate[sample];
        data[sample_index] = data_candidate[sample];
        posterior[sample_index] = posterior_candidate[sample];
        // set the flag 
        acceptance_flag[sample] = 1;
    }
}

// end of file
