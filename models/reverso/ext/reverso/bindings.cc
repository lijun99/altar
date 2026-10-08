// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include <array>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../../lib/libreverso/reverso.h"

#include "bindings.h"

namespace py = pybind11;
using namespace py::literals;

namespace reverso = altar::models::reverso;
using layout_t = std::array<std::size_t, reverso::PARAMETERS>;

namespace {
    // the numpy arrays the bindings take: doubles, never converted (the arguments are
    // {noconvert}), so that results land in them
    using array_t = py::array_t<double, 0>;
    constexpr py::ssize_t cell = sizeof(double);

    // views of their memory, in cells
    auto input(const array_t & a, const char * name) -> reverso::const_matrix_t
    {
        if (a.ndim() != 2) {
            throw py::value_error(std::string(name) + " must be two dimensional");
        }
        return { a.data(), std::size_t(a.shape(0)), std::size_t(a.shape(1)),
                 a.strides(0) / cell, a.strides(1) / cell };
    }

    auto output(array_t & a, const char * name) -> reverso::matrix_t
    {
        if (a.ndim() != 2) {
            throw py::value_error(std::string(name) + " must be two dimensional");
        }
        return { a.mutable_data(), std::size_t(a.shape(0)), std::size_t(a.shape(1)),
                 a.strides(0) / cell, a.strides(1) / cell };
    }

    auto flags(array_t & a, const char * name) -> reverso::vector_t
    {
        if (a.ndim() != 1) {
            throw py::value_error(std::string(name) + " must be one dimensional");
        }
        return { a.mutable_data(), std::size_t(a.shape(0)), a.strides(0) / cell };
    }
}


void
altar::models::reverso::extension::bindings(pybind11::module & m)
{
    m.def(
        "displacements",
        [](const array_t & thetaArray, const array_t & stationsArray, const layout_t & layout,
           double G, double v, double mu, double drho, double g,
           bool shallowSill, bool deepSill, std::size_t batch, array_t & predictedArray) -> void {
            auto theta = input(thetaArray, "reverso.displacements: theta");
            auto stations = input(stationsArray, "reverso.displacements: stations");
            auto predicted = output(predictedArray, "reverso.displacements: predicted");
            if (stations.cols != reverso::STATION_COLUMNS) {
                throw py::value_error("reverso.displacements: stations must have 3 columns");
            }
            if (batch > theta.rows || batch > predicted.rows
                || predicted.cols != 3 * stations.rows) {
                throw py::value_error("reverso.displacements: mismatched theta/predicted shapes");
            }
            reverso::medium_t medium { G, v, mu, drho, g, shallowSill, deepSill };
            reverso::displacements(theta, stations, layout.data(), medium, batch, predicted);
        },
        "theta"_a.noconvert(), "stations"_a.noconvert(), "layout"_a, "G"_a, "v"_a, "mu"_a, "drho"_a, "g"_a,
        "shallowSill"_a, "deepSill"_a, "batch"_a, "predicted"_a.noconvert(),
        "fill predicted[:batch] with the (east, north, up) displacements of the models in theta[:batch]");

    m.def(
        "verify",
        [](const array_t & thetaArray, const layout_t & layout, std::size_t batch,
           array_t & maskArray) -> void {
            auto theta = input(thetaArray, "reverso.verify: theta");
            auto mask = flags(maskArray, "reverso.verify: mask");
            if (batch > theta.rows || batch > mask.size) {
                throw py::value_error("reverso.verify: mismatched theta/mask shapes");
            }
            reverso::verify(theta, layout.data(), batch, mask);
        },
        "theta"_a.noconvert(), "layout"_a, "batch"_a, "mask"_a.noconvert(),
        "flag in mask the samples in theta[:batch] whose deep chamber isn't below the shallow one");
}

// end of file
