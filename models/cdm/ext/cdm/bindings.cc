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
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../../lib/libcdm/cdm.h"

#include "bindings.h"

namespace py = pybind11;
using namespace py::literals;

namespace cdm = altar::models::cdm;
using layout_t = std::array<std::size_t, cdm::PARAMETERS>;

namespace {
    // the numpy arrays the bindings take: doubles, never converted (the arguments are
    // {noconvert}), so that results land in them
    using array_t = py::array_t<double, 0>;
    constexpr py::ssize_t cell = sizeof(double);

    // views of their memory, in cells
    auto input(const array_t & a, const char * name) -> cdm::const_matrix_t
    {
        if (a.ndim() != 2) {
            throw py::value_error(std::string(name) + " must be two dimensional");
        }
        return { a.data(), std::size_t(a.shape(0)), std::size_t(a.shape(1)),
                 a.strides(0) / cell, a.strides(1) / cell };
    }

    auto output(array_t & a, const char * name) -> cdm::matrix_t
    {
        if (a.ndim() != 2) {
            throw py::value_error(std::string(name) + " must be two dimensional");
        }
        return { a.mutable_data(), std::size_t(a.shape(0)), std::size_t(a.shape(1)),
                 a.strides(0) / cell, a.strides(1) / cell };
    }

    auto flags(array_t & a, const char * name) -> cdm::vector_t
    {
        if (a.ndim() != 1) {
            throw py::value_error(std::string(name) + " must be one dimensional");
        }
        return { a.mutable_data(), std::size_t(a.shape(0)), a.strides(0) / cell };
    }
}


void
altar::models::cdm::extension::bindings(pybind11::module & m)
{
    m.def(
        "displacements",
        [](const array_t & thetaArray, const array_t & stationsArray, const layout_t & layout,
           double nu, std::size_t batch, array_t & predictedArray) -> void {
            auto theta = input(thetaArray, "cdm.displacements: theta");
            auto stations = input(stationsArray, "cdm.displacements: stations");
            auto predicted = output(predictedArray, "cdm.displacements: predicted");
            if (stations.cols != cdm::STATION_COLUMNS) {
                throw py::value_error("cdm.displacements: stations must have 6 columns");
            }
            if (batch > theta.rows || batch > predicted.rows || predicted.cols != stations.rows) {
                throw py::value_error("cdm.displacements: mismatched theta/predicted shapes");
            }
            cdm::displacements(theta, stations, layout.data(), nu, batch, predicted);
        },
        "theta"_a.noconvert(), "stations"_a.noconvert(), "layout"_a, "nu"_a, "batch"_a, "predicted"_a.noconvert(),
        "fill predicted[:batch] with the LOS displacements of the CDM sources in theta[:batch]");

    m.def(
        "verify",
        [](const array_t & thetaArray, const layout_t & layout, std::size_t batch,
           array_t & maskArray) -> void {
            auto theta = input(thetaArray, "cdm.verify: theta");
            auto mask = flags(maskArray, "cdm.verify: mask");
            if (batch > theta.rows || batch > mask.size) {
                throw py::value_error("cdm.verify: mismatched theta/mask shapes");
            }
            cdm::verify(theta, layout.data(), batch, mask);
        },
        "theta"_a.noconvert(), "layout"_a, "batch"_a, "mask"_a.noconvert(),
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
