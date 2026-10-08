// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#pragma once

// the two magma chamber model of Reverso et al. [2014], JGR 119, 4666-4683: a deep chamber fed
// at a constant rate {Qin} feeds a shallow one through a cylindrical conduit; the analytic
// overpressures of the two, starting from zero, drive the surface displacements of two point
// sources, sills or spheres, under the origin

#include <cstddef>

namespace altar::models::reverso {

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

    // the columns of a {stations} matrix: the time and the location of each observation
    enum station_t { T = 0, X, Y, STATION_COLUMNS };

    // the columns of the model parameters in {theta}: the basal inflow rate, the depths and
    // radii of the shallow and deep chambers, and the radius of the conduit
    enum parameter_t { QIN = 0, H_S, H_D, A_S, A_D, A_C, PARAMETERS };

    // the medium: shear modulus, poisson ratio, magma viscosity, the density contrast of the
    // rock and the magma, gravity, and whether each chamber is a sill or a sphere
    struct medium_t {
        double G, v, mu, drho, g;
        bool shallowSill, deepSill;
    };

    // fill the first {batch} rows of {predicted} (samples x 3 observations) with the (east,
    // north, up) displacements at each observation of the models in {theta}
    void displacements(const const_matrix_t & theta, const const_matrix_t & stations,
                       const std::size_t * layout, const medium_t & medium, std::size_t batch,
                       const matrix_t & predicted);

    // flag in {mask} the first {batch} samples whose deep chamber isn't below the shallow one
    void verify(const const_matrix_t & theta, const std::size_t * layout, std::size_t batch,
                const vector_t & mask);

} // of namespace altar::models::reverso

// end of file
