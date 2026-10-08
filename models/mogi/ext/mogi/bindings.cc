// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../../lib/libmogi/mogi.h"

#include "bindings.h"

namespace py = pybind11;
using namespace py::literals;

namespace {
    // the numpy arrays the bindings take: doubles, never converted (the arguments are
    // {noconvert}), so that results land in them
    using array_t = py::array_t<double, 0>;
    constexpr py::ssize_t cell = sizeof(double);

    // views of their memory, in cells
    auto input(const array_t & a, const char * name) -> altar::models::mogi::const_matrix_t
    {
        if (a.ndim() != 2) {
            throw py::value_error(std::string(name) + " must be two dimensional");
        }
        return { a.data(), std::size_t(a.shape(0)), std::size_t(a.shape(1)),
                 a.strides(0) / cell, a.strides(1) / cell };
    }

    auto output(array_t & a, const char * name) -> altar::models::mogi::matrix_t
    {
        if (a.ndim() != 2) {
            throw py::value_error(std::string(name) + " must be two dimensional");
        }
        return { a.mutable_data(), std::size_t(a.shape(0)), std::size_t(a.shape(1)),
                 a.strides(0) / cell, a.strides(1) / cell };
    }

    auto flags(array_t & a, const char * name) -> altar::models::mogi::vector_t
    {
        if (a.ndim() != 1) {
            throw py::value_error(std::string(name) + " must be one dimensional");
        }
        return { a.mutable_data(), std::size_t(a.shape(0)), a.strides(0) / cell };
    }
}


void
altar::models::mogi::extension::bindings(pybind11::module & m)
{
    m.def(
        "displacements",
        [](const array_t & thetaArray, const array_t & stationsArray,
           std::size_t xIdx, std::size_t yIdx, std::size_t dIdx, std::size_t sIdx,
           bool log10dV, double nu, std::size_t batch, array_t & predictedArray) -> void {
            auto theta = input(thetaArray, "mogi.displacements: theta");
            auto stations = input(stationsArray, "mogi.displacements: stations");
            auto predicted = output(predictedArray, "mogi.displacements: predicted");
            if (stations.cols != altar::models::mogi::STATION_COLUMNS) {
                throw py::value_error("mogi.displacements: stations must have 6 columns");
            }
            if (batch > theta.rows || batch > predicted.rows || predicted.cols != stations.rows) {
                throw py::value_error("mogi.displacements: mismatched theta/predicted shapes");
            }
            altar::models::mogi::displacements(
                theta, stations, xIdx, yIdx, dIdx, sIdx, log10dV, nu, batch, predicted);
        },
        "theta"_a.noconvert(), "stations"_a.noconvert(), "xIdx"_a, "yIdx"_a, "dIdx"_a, "sIdx"_a,
        "log10dV"_a, "nu"_a, "batch"_a, "predicted"_a.noconvert(),
        "fill predicted[:batch] with the LOS displacements of the Mogi sources in theta[:batch]");
}

// end of file
