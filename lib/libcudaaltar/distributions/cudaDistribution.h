// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2020 parasim inc
// (c) 2010-2020 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

/// altar/cuda/distributions/cudaDistribution.h
/// Abstract class for cuda distributions

// code guard
#ifndef altar_cuda_distributions_cudaDistribution_h
#define altar_cuda_distributions_cudaDistribution_h

// place everything in the local namespace
namespace altar { namespace cuda{
    namespace distributions {
        class cudaDistribution;
    } // of namespace distributions
} }// of namespace cualtar


// Samples are stored in theta (samples, parameters) with leading dimension = parameters
// A range of parameters [idx_begin, idx_end) 

class altar::cuda::distributions::cudaDistribution 
{
// local variables
protected: 
    size_t _idx_begin; // start index 
    size_t _idx_end; // end index 

// methods 
public:
    // initialize random samples, must be defined by child class
    virtual void sample(double * const theta, const size_t samples, const size_t parameters, cudaStream_t stream=0)=0;
    // verify the validity of each sample, do nothing by default 
    virtual void verify(const double * const theta, int * const invalid, const size_t samples, const size_t parameters, cudaStream_t stream=0) {return;}
    // compute the sum of log pdf for each sample, must de defined by child class
    virtual void logpdf(const double * const theta, double * const probability, const size_t samples, const size_t parameters, cudaStream_t stream=0)=0;
    
// meta-methods
public:
    cudaDistribution(size_t idx_begin, size_t idx_end) : _idx_begin(idx_begin), _idx_end(idx_end) {}
    ~cudaDistribution() {}
    
// disallow copy and move    
public:
    cudaDistribution(const cudaDistribution&) =delete;
    cudaDistribution& operator=(const cudaDistribution &) =delete;
    cudaDistribution(cudaDistribution&&) = delete;
    cudaDistribution& operator=(cudaDistribution&&) =delete; 
};

#endif //altar_cuda_distributions_cudaDistribution_h
// end of file
