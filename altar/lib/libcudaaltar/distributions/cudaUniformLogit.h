// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s):  Lijun Zhu

// code guard
#ifndef altar_cuda_distributions_cudaUniformLogit_h
#define altar_cuda_distributions_cudaUniformLogit_h

#include <cuda_runtime.h> // for cudaStream_t
#include <curand_kernel.h> // for curandState_t

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace distributions {
            namespace cudaUniformLogit {

                //! Sample in sampling space (phi)
                //! @note: Use cudaLogistic::sample instead - identical sampling process
                //! for generating logit-transformed uniform random numbers

                //! Transform physical variables to sampling variables using logit function
                //! Maps [low, high] to (-∞, ∞)
                template <typename real_type>
                void tosampling(const real_type * const theta, real_type * const phi,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                //! Transform sampling variables to physical variables using sigmoid/expit function
                //! Maps (-∞, ∞) to [low, high]
                template <typename real_type>
                void tophysical(const real_type * const phi, real_type * const theta,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                //! Compute log PDF P(phi)
                //! Uses physical parameters (theta) for efficiency
                template <typename real_type>
                void logpdf(const real_type * const theta,
                    real_type * const probability,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                //! Compute gradient of log PDF d\log P(phi)/d\phi
                //! Uses physical parameters (theta) for efficiency
                template <typename real_type>
                void logpdfgradient(const real_type * const theta,
                    real_type * const gradient,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

                //! Compute Jacobian matrix d\theta/d\phi
                //! Uses physical parameters (theta) for efficiency
                template <typename real_type>
                void jacobian(const real_type * const theta,
                    real_type * const jacobian,
                    const size_t samples, const size_t parameters,
                    const size_t idx_begin, const size_t idx_end,
                    const real_type low, const real_type high,
                    cudaStream_t stream=0);

            }  // namespace cudaUniformLogit
        }  // namespace distributions
    }  // namespace cuda
}  // namespace altar

#endif
