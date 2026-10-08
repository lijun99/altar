// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include <array>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../lib/libreverso/reverso.h"

namespace py = pybind11;
using namespace py::literals;

namespace reverso = altar::models::reverso;
using layout_t = std::array<std::size_t, reverso::PARAMETERS>;


PYBIND11_MODULE(reverso, m)
{
    m.doc() = "the altar reverso extension module";

    m.def(
        "displacements",
        [](const gsl_matrix & theta, const gsl_matrix & stations, const layout_t & layout,
           double G, double v, double mu, double drho, double g,
           bool shallowSill, bool deepSill, std::size_t batch, gsl_matrix & predicted) -> void {
            if (stations.size2 != reverso::STATION_COLUMNS) {
                throw py::value_error("reverso.displacements: stations must have 3 columns");
            }
            if (batch > theta.size1 || batch > predicted.size1
                || predicted.size2 != 3 * stations.size1) {
                throw py::value_error("reverso.displacements: mismatched theta/predicted shapes");
            }
            reverso::medium_t medium { G, v, mu, drho, g, shallowSill, deepSill };
            reverso::displacements(theta, stations, layout.data(), medium, batch, predicted);
        },
        "theta"_a, "stations"_a, "layout"_a, "G"_a, "v"_a, "mu"_a, "drho"_a, "g"_a,
        "shallowSill"_a, "deepSill"_a, "batch"_a, "predicted"_a,
        "fill predicted[:batch] with the (east, north, up) displacements of the models in theta[:batch]");

    m.def(
        "verify",
        [](const gsl_matrix & theta, const layout_t & layout, std::size_t batch,
           gsl_vector & mask) -> void {
            if (batch > theta.size1 || batch > mask.size) {
                throw py::value_error("reverso.verify: mismatched theta/mask shapes");
            }
            reverso::verify(theta, layout.data(), batch, mask);
        },
        "theta"_a, "layout"_a, "batch"_a, "mask"_a,
        "flag in mask the samples in theta[:batch] whose deep chamber isn't below the shallow one");
}

// end of file
