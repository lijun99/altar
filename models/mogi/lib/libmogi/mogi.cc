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
displacements(const const_matrix_t & theta, const const_matrix_t & stations,
              std::size_t xIdx, std::size_t yIdx, std::size_t dIdx, std::size_t sIdx,
              bool log10dV, double nu, std::size_t batch, const matrix_t & predicted)
{
    for (std::size_t sample = 0; sample < batch; ++sample) {
        // the source of this sample
        auto xSrc = theta(sample, xIdx);
        auto ySrc = theta(sample, yIdx);
        auto dSrc = theta(sample, dIdx);
        auto s = theta(sample, sIdx);
        auto dV = log10dV ? std::pow(10.0, s) : s;

        for (std::size_t obs = 0; obs < stations.rows; ++obs) {
            auto u = los(xSrc, ySrc, dSrc, dV, nu,
                         stations(obs, X), stations(obs, Y),
                         stations(obs, LOS_E),
                         stations(obs, LOS_N),
                         stations(obs, LOS_U));
            // shift by the offset of the observation's dataset, if any
            auto offset = stations(obs, OFFSET);
            if (offset >= 0) {
                u -= theta(sample, static_cast<std::size_t>(offset));
            }
            predicted(sample, obs) = u;
        }
    }
}

// end of file
