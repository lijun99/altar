// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// the cpu kinematic models, on numpy arrays: the rectangular patches by fast sweeping, and the
// triangulated fault by the fast iterative method; each at the precision it was made with

#include <memory>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../../lib/libseismic/Kinematic.h"
#include "../../lib/libseismic/KinematicMesh.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace {
    using ints_t = py::array_t<int, py::array::c_style | py::array::forcecast>;
    using doubles_t = py::array_t<double, py::array::c_style | py::array::forcecast>;
    using sizes_t = py::array_t<std::size_t, py::array::c_style | py::array::forcecast>;

    // the cells of {a}, a c-contiguous (rows x columns) array of {T}, at least {rows} long;
    // writable if {output}, so a result never lands in a converted copy
    template <typename T>
    auto cells(py::array & a, std::size_t rows, std::size_t columns, bool output, const char * name) -> T *
    {
        if (!a.dtype().is(py::dtype::of<T>())) {
            throw py::type_error(std::string(name) + ": the precision differs from the model's");
        }
        if (a.ndim() != 2 || std::size_t(a.shape(1)) != columns || std::size_t(a.shape(0)) < rows
            || !(a.flags() & py::array::c_style)) {
            throw py::value_error(std::string(name) + ": needs a c-contiguous array of "
                                  + std::to_string(columns) + " columns and at least "
                                  + std::to_string(rows) + " rows");
        }
        if (output && !a.writeable()) {
            throw py::value_error(std::string(name) + ": the array is read-only");
        }
        return static_cast<T *>(a.mutable_data());
    }

    // a model at either precision, {M} the class template
    template <template <typename> class M>
    class Model {
    public:
        // Mb <- the slips of each patch over time, for the first {batch} rows of {theta}
        auto cast_mb(py::array theta, py::array mb, std::size_t batch) -> void
        {
            dispatch([&](auto & model, auto * zero) {
                using T = std::remove_pointer_t<decltype(zero)>;
                auto parameters = std::size_t(theta.shape(1));
                auto t = cells<T>(theta, batch, parameters, false, "cast_mb");
                auto m = cells<T>(mb, batch, _NGb, true, "cast_mb");
                py::gil_scoped_release release;
                model.calculateBigM(t, m, parameters, batch);
            });
        }

        // gradient <- d(log L)/d(theta) on my columns, from dmb = d(log L)/d(Mb)
        auto gradient(py::array theta, py::array dmb, py::array gradient, std::size_t batch) -> void
        {
            dispatch([&](auto & model, auto * zero) {
                using T = std::remove_pointer_t<decltype(zero)>;
                auto parameters = std::size_t(theta.shape(1));
                auto t = cells<T>(theta, batch, parameters, false, "gradient");
                auto d = cells<T>(dmb, batch, _NGb, false, "gradient");
                auto g = cells<T>(gradient, batch, parameters, true, "gradient");
                py::gil_scoped_release release;
                model.gradient(t, d, g, parameters, batch);
            });
        }

        // the arrival times of the first {batch} rows of {theta}, on the mesh
        auto arrivals(py::array theta, std::size_t batch) -> py::array
        {
            py::array out;
            dispatch([&](auto & model, auto * zero) {
                using T = std::remove_pointer_t<decltype(zero)>;
                auto parameters = std::size_t(theta.shape(1));
                auto t = cells<T>(theta, batch, parameters, false, "arrivals");
                auto T_ = py::array_t<T>(std::vector<py::ssize_t>{py::ssize_t(batch), py::ssize_t(_cells)});
                auto a = T_.mutable_data();
                {
                    py::gil_scoped_release release;
                    model.arrivals(t, a, parameters, batch);
                }
                out = T_;
            });
            return out;
        }

    protected:
        template <typename F>
        auto dispatch(F && f) -> void
        {
            if (_double) f(*_double, static_cast<double *>(nullptr));
            else f(*_float, static_cast<float *>(nullptr));
        }

        std::unique_ptr<M<double>> _double;
        std::unique_ptr<M<float>> _float;
        std::size_t _NGb = 0, _cells = 0;
    };

    // the rectangular patches
    class Kinematic : public Model<altar::models::seismic::Kinematic> {
    public:
        Kinematic(std::size_t Nas, std::size_t Ndd, std::size_t Nmesh, double dsp, std::size_t Nt,
                  std::size_t Npt, double dt, doubles_t t0s, sizes_t idx_map, const std::string & precision)
        {
            std::size_t Np = Nas*Ndd;
            if (std::size_t(t0s.size()) != Np || std::size_t(idx_map.size()) != 4*Np + 2) {
                throw py::value_error("Kinematic: t0s needs a value a patch, idx_map 4 a patch plus 2");
            }
            auto make = [&](auto * zero) {
                using T = std::remove_pointer_t<decltype(zero)>;
                return std::make_unique<altar::models::seismic::Kinematic<T>>(
                    Nas, Ndd, Nmesh, dsp, Nt, Npt, dt, t0s.data(), idx_map.data());
            };
            if (precision == "float64") _double = make(static_cast<double *>(nullptr));
            else if (precision == "float32") _float = make(static_cast<float *>(nullptr));
            else throw py::value_error("Kinematic: precision is float64 or float32, not '" + precision + "'");
            _NGb = 2*Np*Nt;
            _cells = (Nas+2)*Nmesh*(Ndd+2)*Nmesh;
        }
    };

    // the triangulated fault
    class KinematicMesh : public Model<altar::models::seismic::KinematicMesh> {
    public:
        KinematicMesh(std::size_t Np, std::size_t Nt, double dt, doubles_t t0s, doubles_t xyz,
                      ints_t incPtr, ints_t incA, ints_t incB, ints_t incPatch, ints_t nbPtr, ints_t nb,
                      ints_t vpPtr, ints_t vpPatch, doubles_t patchGeometry, double spacing,
                      std::size_t Nsp, ints_t corners, doubles_t weights, sizes_t idx_map,
                      const std::string & precision)
        {
            std::size_t V = xyz.size()/3, P = corners.size()/3;
            if (std::size_t(incPtr.size()) != V + 1 || std::size_t(nbPtr.size()) != V + 1
                || std::size_t(vpPtr.size()) != V + 1 || std::size_t(patchGeometry.size()) != 15*Np
                || std::size_t(weights.size()) != 3*P || P != Np*Nsp || std::size_t(t0s.size()) != Np
                || std::size_t(idx_map.size()) != 4*Np + 2) {
                throw py::value_error("KinematicMesh: inconsistent mesh tables");
            }
            auto make = [&](auto * zero) {
                using T = std::remove_pointer_t<decltype(zero)>;
                return std::make_unique<altar::models::seismic::KinematicMesh<T>>(
                    Np, Nt, dt, t0s.data(), V, xyz.data(), incA.size(), incPtr.data(), incA.data(),
                    incB.data(), incPatch.data(), nb.size(), nbPtr.data(), nb.data(), vpPatch.size(),
                    vpPtr.data(), vpPatch.data(), patchGeometry.data(), spacing, P, Nsp,
                    corners.data(), weights.data(), idx_map.data());
            };
            if (precision == "float64") _double = make(static_cast<double *>(nullptr));
            else if (precision == "float32") _float = make(static_cast<float *>(nullptr));
            else throw py::value_error("KinematicMesh: precision is float64 or float32, not '" + precision + "'");
            _NGb = 2*Np*Nt;
            _cells = V;
        }

        // the most passes of the eikonal over the last {batch} samples
        auto iterations(std::size_t batch) -> int
        {
            return _double ? _double->iterations(batch) : _float->iterations(batch);
        }
    };
}


PYBIND11_MODULE(seismic, m)
{
    m.doc() = "the cpu kinematic slip models";

    py::class_<Kinematic>(m, "Kinematic",
                          "the kinematic slip model of rectangular patches, by fast sweeping")
        .def(py::init<std::size_t, std::size_t, std::size_t, double, std::size_t, std::size_t, double,
                      doubles_t, sizes_t, const std::string &>(),
             "Nas"_a, "Ndd"_a, "Nmesh"_a, "dsp"_a, "Nt"_a, "Npt"_a, "dt"_a, "t0s"_a, "idx_map"_a,
             "precision"_a)
        .def("cast_mb", &Kinematic::cast_mb, "theta"_a, "mb"_a, "batch"_a,
             "mb <- the slips of each patch over time, from {theta}")
        .def("gradient", &Kinematic::gradient, "theta"_a, "dmb"_a, "gradient"_a, "batch"_a,
             "gradient <- d(log L)/d(theta) on my columns, from dmb = d(log L)/d(Mb)")
        .def("arrivals", &Kinematic::arrivals, "theta"_a, "batch"_a,
             "the arrival times on the padded mesh, (batch x cells)");

    py::class_<KinematicMesh>(m, "KinematicMesh",
                              "the kinematic slip model on a triangulated fault, by the fast iterative method")
        .def(py::init<std::size_t, std::size_t, double, doubles_t, doubles_t, ints_t, ints_t, ints_t,
                      ints_t, ints_t, ints_t, ints_t, ints_t, doubles_t, double, std::size_t, ints_t,
                      doubles_t, sizes_t, const std::string &>(),
             "Np"_a, "Nt"_a, "dt"_a, "t0s"_a, "xyz"_a, "incPtr"_a, "incA"_a, "incB"_a, "incPatch"_a,
             "nbPtr"_a, "nb"_a, "vpPtr"_a, "vpPatch"_a, "patchGeometry"_a, "spacing"_a, "Nsp"_a,
             "corners"_a, "weights"_a, "idx_map"_a, "precision"_a)
        .def("cast_mb", &KinematicMesh::cast_mb, "theta"_a, "mb"_a, "batch"_a,
             "mb <- the slips of each patch over time, from {theta}")
        .def("gradient", &KinematicMesh::gradient, "theta"_a, "dmb"_a, "gradient"_a, "batch"_a,
             "gradient <- d(log L)/d(theta) on my columns, from dmb = d(log L)/d(Mb)")
        .def("arrivals", &KinematicMesh::arrivals, "theta"_a, "batch"_a,
             "the arrival times at the vertices, (batch x vertices)")
        .def("iterations", &KinematicMesh::iterations, "batch"_a,
             "the most passes of the eikonal over the last {batch} samples");
}

// end of file
