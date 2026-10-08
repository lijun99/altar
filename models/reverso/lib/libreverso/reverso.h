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
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

namespace altar::models::reverso {

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
    void displacements(const gsl_matrix & theta, const gsl_matrix & stations,
                       const std::size_t * layout, const medium_t & medium, std::size_t batch,
                       gsl_matrix & predicted);

    // flag in {mask} the first {batch} samples whose deep chamber isn't below the shallow one
    void verify(const gsl_matrix & theta, const std::size_t * layout, std::size_t batch,
                gsl_vector & mask);

} // of namespace altar::models::reverso

// end of file
