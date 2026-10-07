// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#pragma once

#include <cmath>
#include <cstddef>
#include <gsl/gsl_matrix.h>

// the point source formula is shared by the cpu and cuda builds
#if defined(__CUDACC__)
#define ALTAR_MOGI_HD __host__ __device__
#else
#define ALTAR_MOGI_HD
#endif

namespace altar::models::mogi {

    // the columns of a {stations} matrix: location, LOS unit vector (east, north, up), and the
    // column of the observation's dataset offset in {theta}, or -1 for none
    enum station_t { X = 0, Y, LOS_E, LOS_N, LOS_U, OFFSET, STATION_COLUMNS };

    // the displacement along the LOS unit vector (nE, nN, nU) at (xObs, yObs) on the surface of
    // an elastic half space, of a point source at (xSrc, ySrc, depth) with volume change {dV}
    template <typename real_t>
    ALTAR_MOGI_HD inline auto
    los(real_t xSrc, real_t ySrc, real_t depth, real_t dV, real_t nu,
        real_t xObs, real_t yObs, real_t nE, real_t nN, real_t nU) -> real_t
    {
        using std::sqrt;
        const auto pi = real_t(3.14159265358979323846);
        auto x = xObs - xSrc;
        auto y = yObs - ySrc;
        auto R2 = x*x + y*y + depth*depth;
        auto C = (1 - nu) * dV / (pi * R2 * sqrt(R2));
        return C * (x*nE + y*nN + depth*nU);
    }

    // fill the first {batch} rows of {predicted} (samples x observations) with the LOS
    // displacements, less the dataset offsets, of the Mogi sources in {theta}
    void displacements(const gsl_matrix & theta, const gsl_matrix & stations,
                       std::size_t xIdx, std::size_t yIdx, std::size_t dIdx, std::size_t sIdx,
                       bool log10dV, double nu, std::size_t batch, gsl_matrix & predicted);

} // of namespace altar::models::mogi

// end of file
