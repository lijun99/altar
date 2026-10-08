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

namespace altar::models::mogi {

    // a strided view of a matrix, e.g. of a numpy array, in cells
    template <typename cell_t>
    struct matrix_view_t {
        cell_t * data;
        std::size_t rows, cols;
        std::ptrdiff_t rowStride, colStride;

        auto operator()(std::size_t row, std::size_t col) const -> cell_t & {
            return data[static_cast<std::ptrdiff_t>(row) * rowStride
                        + static_cast<std::ptrdiff_t>(col) * colStride];
        }
    };

    // and of a vector
    template <typename cell_t>
    struct vector_view_t {
        cell_t * data;
        std::size_t size;
        std::ptrdiff_t stride;

        auto operator[](std::size_t i) const -> cell_t & {
            return data[static_cast<std::ptrdiff_t>(i) * stride];
        }
    };

    using const_matrix_t = matrix_view_t<const double>;
    using matrix_t = matrix_view_t<double>;
    using vector_t = vector_view_t<double>;

    // the columns of a {stations} matrix: location, LOS unit vector (east, north, up), and the
    // column of the observation's dataset offset in {theta}, or -1 for none
    enum station_t { X = 0, Y, LOS_E, LOS_N, LOS_U, OFFSET, STATION_COLUMNS };

    // the displacement along the LOS unit vector (nE, nN, nU) at (xObs, yObs) on the surface of
    // an elastic half space, of a point source at (xSrc, ySrc, depth) with volume change {dV}
    template <typename real_t>
    inline auto
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
    void displacements(const const_matrix_t & theta, const const_matrix_t & stations,
                       std::size_t xIdx, std::size_t yIdx, std::size_t dIdx, std::size_t sIdx,
                       bool log10dV, double nu, std::size_t batch, const matrix_t & predicted);

} // of namespace altar::models::mogi

// end of file
