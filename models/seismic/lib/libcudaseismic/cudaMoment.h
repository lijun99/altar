// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#pragma once

#include <cuda_runtime.h>
// {matrix_view_t}/{vector_view_t}
#include <altar/cuda/support.h>

namespace altar::models::seismic::cudaMoment {

    // likelihood[s] += -factor*(Mw - mean)^2/(2 sigma^2), Mw = (log10|M0| + 5.9)/1.5,
    // M0 = sum_i mu_area[i] theta[s, idx_begin+i]; an accumulation
    template <typename real_type>
    void logpdf(matrix_view_t<real_type> theta, vector_view_t<real_type> likelihood,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        vector_view_t<real_type, true> mu_area, const real_type factor,
        cudaStream_t stream=0);

    // gradient[:, idx_begin:idx_end] <- d/dtheta of {logpdf}'s term; an assignment
    template <typename real_type>
    void logpdf_gradient(matrix_view_t<real_type> theta, matrix_view_t<real_type, false> gradient,
        const size_t idx_begin, const size_t idx_end,
        const real_type mean, const real_type sigma,
        vector_view_t<real_type, true> mu_area, const real_type factor,
        cudaStream_t stream=0);

} // of namespace altar::models::seismic::cudaMoment

// end of file
