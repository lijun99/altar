// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// declarations
#include "mogi.h"


void
altar::models::mogi::
displacements(const gsl_matrix & theta, const gsl_matrix & stations,
              std::size_t xIdx, std::size_t yIdx, std::size_t dIdx, std::size_t sIdx,
              bool log10dV, double nu, std::size_t batch, gsl_matrix & predicted)
{
    for (std::size_t sample = 0; sample < batch; ++sample) {
        // the source of this sample
        auto xSrc = gsl_matrix_get(&theta, sample, xIdx);
        auto ySrc = gsl_matrix_get(&theta, sample, yIdx);
        auto dSrc = gsl_matrix_get(&theta, sample, dIdx);
        auto s = gsl_matrix_get(&theta, sample, sIdx);
        auto dV = log10dV ? std::pow(10.0, s) : s;

        for (std::size_t obs = 0; obs < stations.size1; ++obs) {
            auto u = los(xSrc, ySrc, dSrc, dV, nu,
                         gsl_matrix_get(&stations, obs, X), gsl_matrix_get(&stations, obs, Y),
                         gsl_matrix_get(&stations, obs, LOS_E),
                         gsl_matrix_get(&stations, obs, LOS_N),
                         gsl_matrix_get(&stations, obs, LOS_U));
            // shift by the offset of the observation's dataset, if any
            auto offset = gsl_matrix_get(&stations, obs, OFFSET);
            if (offset >= 0) {
                u -= gsl_matrix_get(&theta, sample, static_cast<std::size_t>(offset));
            }
            gsl_matrix_set(&predicted, sample, obs, u);
        }
    }
}

// end of file
