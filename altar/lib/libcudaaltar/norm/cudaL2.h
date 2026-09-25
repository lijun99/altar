// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-2025 parasim inc
// (c) 2010-2025 california institute of technology
// all rights reserved
//
// Author(s): Hailiang Zhang, Lijun Zhu

// code guard
#ifndef altar_cuda_norms_cudaL2_h
#define altar_cuda_norms_cudaL2_h

#include <cuda_runtime.h>

// pyre's own statically typed grid: {WITH_CUDA} (set for this whole library, see
// altar_cudaframework.cmake) turns on the {PYRE_HOST_DEVICE} decorations these need to be
// callable from a kernel; {--expt-relaxed-constexpr} (also set there) lets device code reach
// the constexpr machinery under {Shape}/{Index}
#include <pyre/grid.h>
#include <pyre/memory.h>

// place everything in the local namespace
namespace altar {
    namespace cuda {
        namespace norms {
            namespace cudaL2 {

                // a (samples x parameters) view over someone else's cells, read-only; the
                // memory itself is owned by whoever built the view (typically a python
                // {pyre.grid.managed} buffer, reconstructed here via {altar::cuda::extensions::
                // regrid}), so neither this alias nor the functions below allocate or free
                template <typename real_type>
                using data_view_t = pyre::grid::grid_t<
                    pyre::grid::canonical_t<2>, pyre::memory::View<real_type, true>>;

                // a (samples,) view over someone else's cells, writable; same ownership note
                // as {data_view_t}
                template <typename real_type>
                using result_view_t = pyre::grid::grid_t<
                    pyre::grid::canonical_t<1>, pyre::memory::View<real_type, false>>;

                // compute the l2 norm of a batch of {data} (samples x parameters), one row at
                // a time: {probability[s] = ||data[s, :]||} for each of the first {batch}
                // samples ({batch <= data}'s own sample count); {probability} must already be
                // sized to hold at least {batch} entries. {stream} is the cuda stream to
                // launch on, the default stream if not given; the launch is asynchronous, so
                // the caller is responsible for synchronizing before reading {probability}
                template <typename real_type>
                void norm(data_view_t<real_type> data, result_view_t<real_type> probability,
                    const size_t batch, cudaStream_t stream=0);

                // compute the l2 log likelihood of a batch of {data} directly, without a
                // separate {norm} pass: {probability[s] = constant - 0.5 * ||data[s, :]||^2}
                // for each of the first {batch} samples; same shapes, ownership, and
                // asynchronous-launch caveat as {norm}
                template <typename real_type>
                void normllk(data_view_t<real_type> data, result_view_t<real_type> probability,
                    const size_t batch, const real_type constant=0.0, cudaStream_t stream=0);

            } // of namespace cudaL2
        } // of namespace norms
    } // of namespace cuda
} // of namespace altar

#endif //altar_cuda_norms_cudaL2_h
// end of file
