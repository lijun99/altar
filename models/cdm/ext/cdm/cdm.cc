// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include <array>
#include <tuple>
#include <vector>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../lib/libcdm/cdm.h"

namespace py = pybind11;
using namespace py::literals;

namespace cdm = altar::models::cdm;
using layout_t = std::array<std::size_t, cdm::PARAMETERS>;


PYBIND11_MODULE(cdm, m)
{
    m.doc() = "the altar cdm extension module";

    m.def(
        "displacements",
        [](const gsl_matrix & theta, const gsl_matrix & stations, const layout_t & layout,
           double nu, std::size_t batch, gsl_matrix & predicted) -> void {
            if (stations.size2 != cdm::STATION_COLUMNS) {
                throw py::value_error("cdm.displacements: stations must have 6 columns");
            }
            if (batch > theta.size1 || batch > predicted.size1
                || predicted.size2 != stations.size1) {
                throw py::value_error("cdm.displacements: mismatched theta/predicted shapes");
            }
            cdm::displacements(theta, stations, layout.data(), nu, batch, predicted);
        },
        "theta"_a, "stations"_a, "layout"_a, "nu"_a, "batch"_a, "predicted"_a,
        "fill predicted[:batch] with the LOS displacements of the CDM sources in theta[:batch]");

    m.def(
        "verify",
        [](const gsl_matrix & theta, const layout_t & layout, std::size_t batch,
           gsl_vector & mask) -> void {
            if (batch > theta.size1 || batch > mask.size) {
                throw py::value_error("cdm.verify: mismatched theta/mask shapes");
            }
            cdm::verify(theta, layout.data(), batch, mask);
        },
        "theta"_a, "layout"_a, "batch"_a, "mask"_a,
        "flag in mask the samples in theta[:batch] whose source reaches above the free surface");

    m.def(
        "enu",
        [](const std::array<double, cdm::PARAMETERS> & parameters,
           const std::vector<double> & x, const std::vector<double> & y, double nu)
            -> std::vector<std::tuple<double, double, double>> {
            auto s = cdm::source(parameters.data());
            if (!cdm::buried(s)) {
                throw py::value_error("cdm.enu: the source must lie below the free surface");
            }
            std::vector<std::tuple<double, double, double>> u;
            for (std::size_t i = 0; i < x.size(); ++i) {
                auto v = cdm::displacement(s, x[i], y[i], nu);
                u.emplace_back(v.x, v.y, v.z);
            }
            return u;
        },
        "parameters"_a, "x"_a, "y"_a, "nu"_a,
        "the (east, north, up) surface displacements at (x, y) of the CDM source with parameters "
        "(x0, y0, depth, opening, ax, ay, az, omegaX, omegaY, omegaZ)");
}

// end of file
