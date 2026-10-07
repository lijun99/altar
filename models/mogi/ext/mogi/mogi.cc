// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include <gsl/gsl_matrix.h>
#include <pybind11/pybind11.h>

#include "../../lib/libmogi/mogi.h"

namespace py = pybind11;
using namespace py::literals;


PYBIND11_MODULE(mogi, m)
{
    m.doc() = "the altar mogi extension module";

    m.def(
        "displacements",
        [](const gsl_matrix & theta, const gsl_matrix & stations,
           std::size_t xIdx, std::size_t yIdx, std::size_t dIdx, std::size_t sIdx,
           bool log10dV, double nu, std::size_t batch, gsl_matrix & predicted) -> void {
            if (stations.size2 != altar::models::mogi::STATION_COLUMNS) {
                throw py::value_error("mogi.displacements: stations must have 6 columns");
            }
            if (batch > theta.size1 || batch > predicted.size1
                || predicted.size2 != stations.size1) {
                throw py::value_error("mogi.displacements: mismatched theta/predicted shapes");
            }
            altar::models::mogi::displacements(
                theta, stations, xIdx, yIdx, dIdx, sIdx, log10dV, nu, batch, predicted);
        },
        "theta"_a, "stations"_a, "xIdx"_a, "yIdx"_a, "dIdx"_a, "sIdx"_a,
        "log10dV"_a, "nu"_a, "batch"_a, "predicted"_a,
        "fill predicted[:batch] with the LOS displacements of the Mogi sources in theta[:batch]");
}

// end of file
