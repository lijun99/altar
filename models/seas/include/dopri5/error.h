// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// california institute of technology
// (c) 2016-2023  all rights reserved
//

// code guard
#ifndef cuda_oed_error_h
#define cuda_ode_error_h

// externals
#include "external.h"

#include <cstdio>
#include <exception>

#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif

namespace cuda::error {

    // define a custom exception for cuda errors
    struct cudaRuntimeException : public ::std::exception {
        cudaError_t e;
        char str[400];
        cudaRuntimeException(cudaError_t error, const char *s, const char *file, int line)
            : e(error)
        {
            snprintf(str, sizeof(str), "CUDA Error Code %d: %s - %s at %s:%d\n",
                error, cudaGetErrorString(error), s, file, line);
        }
        const char *what() const noexcept override
        {
            return str;
        }
    };

    // define a custom exception for pyre cuda errors
    struct odeException : public ::std::exception {
        char str[400];
        odeException(const char *s, const char *file, int line)
        {
            snprintf(str, sizeof(str), "CUDA ODE Error %s at %s:%d\n",
                s, file, line);
        }
        const char *what() const noexcept override
        {
            return str;
        }
    };
} // end of namespace

//
#ifndef cudaSafeCall
#define cudaSafeCall(x)                                                        \
    {                                                                          \
        if ((x) != cudaSuccess)                                                \
        {                                                                      \
            throw cuda::error::cudaRuntimeException(x, "", __FILE__, __LINE__); \
        }                                                                      \
    }
#endif

// error checking for customized kernels
#ifndef cudaCheckError
#define cudaCheckError(msg)                                                     \
    {                                                                           \
        auto x = cudaGetLastError();                                            \
        if ((x) != cudaSuccess)                                                 \
        {                                                                       \
            throw cuda::error::cudaRuntimeException(x, msg, __FILE__, __LINE__); \
        }                                                                       \
    }
#endif

#ifdef CURAND_H_
  #ifndef curandSafeCall
  #define curandSafeCall(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("CURAND Error %d at %s:%d\n", x, __FILE__,__LINE__); \
      DEVICE_RESET \
      exit(EXIT_FAILURE);}} while(0)
  #endif
#endif // CURAND_H_

#ifdef CUBLAS_API_H_
  #ifndef cublasSafeCall
  #define cublasSafeCall(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
      printf("CUBLAS Error %d at %s:%d\n", x, __FILE__,__LINE__); \
      DEVICE_RESET \
      exit(EXIT_FAILURE);}} while(0)
  #endif
#endif // CUBLAS_API_H_

#endif //cuda_ode_error_h
// end of file
