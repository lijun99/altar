// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// externals
#include <memory>
#include <pybind11/numpy.h>
// my declarations
#include "kinematicmesh.h"
// the model
#include <altar/models/seismic/cuda/cudaKinematicMesh.h>


namespace {
    using namespace altar::cuda::extensions;
    template <typename T> using model_t = altar::models::seismic::cudaKinematicMesh<T>;
    using ints_t = py::array_t<int, py::array::c_style | py::array::forcecast>;
    using doubles_t = py::array_t<double, py::array::c_style | py::array::forcecast>;

    // the cell type of {g}, 'd' or 'f'
    auto cell(grid_t & g) -> char
    {
        auto format = g.view().format;
        if (format.size() != 1 || (format[0] != 'd' && format[0] != 'f')) {
            throw py::value_error("cudaKinematicMesh: unsupported grid cell type '" + format + "'");
        }
        return format[0];
    }

    // the kinematic model on a triangulated fault, at the precision of its {t0s}; the caller keeps
    // {t0s} and {idx_map} alive, the model copies the mesh tables
    class KinematicMesh {
    public:
        KinematicMesh(std::size_t Np, std::size_t Nt, double dt, grid_t & t0s, doubles_t xyz,
                      ints_t incPtr, ints_t incA, ints_t incB, ints_t incPatch, ints_t nbPtr, ints_t nb,
                      ints_t vpPtr, ints_t vpPatch, doubles_t patchGeometry, double spacing,
                      std::size_t Nsp, ints_t corners, doubles_t weights,
                      std::size_t samples, std::size_t parameters, std::size_t observations,
                      grid_t & idx_map) :
            _cell(cell(t0s))
        {
            auto format = idx_map.view().format;
            if (format != "q" && format != "l") {
                throw py::value_error("cudaKinematicMesh: idx_map must be an int64 grid, not '" + format + "'");
            }
            std::size_t V = xyz.size()/3, P = corners.size()/3;
            if (incPtr.size() != V + 1 || nbPtr.size() != V + 1 || vpPtr.size() != V + 1
                || patchGeometry.size() != 15*Np || weights.size() != 3*P || P != Np*Nsp) {
                throw py::value_error("cudaKinematicMesh: inconsistent mesh tables");
            }
            auto idx = reinterpret_cast<const std::size_t *>(idx_map.address());
            auto make = [&](auto * t0) {
                using T = std::remove_const_t<std::remove_pointer_t<decltype(t0)>>;
                return std::make_unique<model_t<T>>(Np, Nt, dt, t0, V, xyz.data(),
                    incA.size(), incPtr.data(), incA.data(), incB.data(), incPatch.data(),
                    nb.size(), nbPtr.data(), nb.data(), vpPatch.size(), vpPtr.data(), vpPatch.data(),
                    patchGeometry.data(), spacing, P, Nsp, corners.data(), weights.data(),
                    samples, parameters, observations, idx);
            };
            if (_cell == 'd') {
                _double = make(reinterpret_cast<const double *>(t0s.address()));
            } else {
                _float = make(reinterpret_cast<const float *>(t0s.address()));
            }
            if (cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cudaKinematicMesh: cublasCreate failed");
            }
        }

        ~KinematicMesh() { cublasDestroy(_handle); }

        // prediction <- Gb Mb(theta), or Gb Mb(theta) - prediction when {residual}
        auto forward_batched(grid_t & theta, grid_t & green, grid_t & prediction,
                             std::size_t batch, bool residual) -> void
        {
            check(theta, green, prediction);
            auto parameters = theta.shape()[1];
            if (_cell == 'd') {
                _double->forwardModel(_handle, ptr<double>(theta), ptr<double>(green), ptr<double>(prediction),
                    parameters, batch, residual);
            } else {
                _float->forwardModel(_handle, ptr<float>(theta), ptr<float>(green), ptr<float>(prediction),
                    parameters, batch, residual);
            }
            done("cudaKinematicMesh.forward_batched");
        }

        // mb <- the slips of each patch over time, from {theta}
        auto cast_mb(grid_t & theta, grid_t & mb, std::size_t batch) -> void
        {
            check(theta, mb, mb);
            auto parameters = theta.shape()[1];
            if (_cell == 'd') {
                _double->calculateBigM(ptr<double>(theta), ptr<double>(mb), parameters, batch);
            } else {
                _float->calculateBigM(ptr<float>(theta), ptr<float>(mb), parameters, batch);
            }
            done("cudaKinematicMesh.cast_mb");
        }

        // prediction <- Gb mb, or Gb mb - prediction when {residual}
        auto linear_gm(grid_t & green, grid_t & mb, grid_t & prediction,
                       std::size_t batch, bool residual) -> void
        {
            check(green, mb, prediction);
            if (_cell == 'd') {
                _double->linearBigGM(_handle, ptr<double>(green), ptr<double>(mb), ptr<double>(prediction),
                    batch, residual);
            } else {
                _float->linearBigGM(_handle, ptr<float>(green), ptr<float>(mb), ptr<float>(prediction),
                    batch, residual);
            }
            done("cudaKinematicMesh.linear_gm");
        }

        // gradient <- d(log L)/d(theta) on my columns, from dmb = d(log L)/d(Mb)
        auto gradient(grid_t & theta, grid_t & dmb, grid_t & gradient, std::size_t batch) -> void
        {
            check(theta, dmb, gradient);
            if (gradient.shape() != theta.shape()) {
                throw py::value_error("cudaKinematicMesh.gradient: the gradient must be shaped like theta");
            }
            auto parameters = theta.shape()[1];
            if (_cell == 'd') {
                _double->gradient(ptr<double>(theta), ptr<double>(dmb), ptr<double>(gradient), parameters, batch);
            } else {
                _float->gradient(ptr<float>(theta), ptr<float>(dmb), ptr<float>(gradient), parameters, batch);
            }
            done("cudaKinematicMesh.gradient");
        }

        // the most passes of the eikonal over the last {batch} samples
        auto iterations(std::size_t batch) -> int
        {
            return _cell == 'd' ? _double->iterations(batch) : _float->iterations(batch);
        }

    private:
        template <typename T>
        static auto ptr(grid_t & g) -> T * { return reinterpret_cast<T *>(g.address()); }

        auto check(grid_t & a, grid_t & b, grid_t & c) const -> void
        {
            if (cell(a) != _cell || cell(b) != _cell || cell(c) != _cell) {
                throw py::value_error("cudaKinematicMesh: grid precision differs from the model's");
            }
        }

        static auto done(const char * routine) -> void
        {
            cudaCheckError(routine);
            synchronize(routine);
        }

        char _cell;
        std::unique_ptr<model_t<double>> _double;
        std::unique_ptr<model_t<float>> _float;
        cublasHandle_t _handle;
    };
}


auto
altar::models::seismic::extensions::kinematicmesh::__init__(py::module & m) -> void
{
    py::class_<KinematicMesh>(m, "cudaKinematicMesh",
                              "the kinematic slip model on a triangulated fault, plus big-G")
        .def(py::init<std::size_t, std::size_t, double, grid_t &, doubles_t, ints_t, ints_t, ints_t,
                      ints_t, ints_t, ints_t, ints_t, ints_t, doubles_t, double, std::size_t, ints_t,
                      doubles_t, std::size_t, std::size_t, std::size_t, grid_t &>(),
             "Np"_a, "Nt"_a, "dt"_a, "t0s"_a, "xyz"_a, "incPtr"_a, "incA"_a, "incB"_a, "incPatch"_a,
             "nbPtr"_a, "nb"_a, "vpPtr"_a, "vpPatch"_a, "patchGeometry"_a, "spacing"_a, "Nsp"_a,
             "corners"_a, "weights"_a, "samples"_a, "parameters"_a, "observations"_a, "idx_map"_a,
             py::keep_alive<1, 5>(), py::keep_alive<1, 23>())
        .def("forward_batched", &KinematicMesh::forward_batched,
             "theta"_a, "green"_a, "prediction"_a, "batch"_a, "residual"_a,
             "prediction <- Gb Mb(theta), or Gb Mb(theta) - prediction when {residual}")
        .def("cast_mb", &KinematicMesh::cast_mb, "theta"_a, "mb"_a, "batch"_a,
             "mb <- the slips of each patch over time, from {theta}")
        .def("gradient", &KinematicMesh::gradient, "theta"_a, "dmb"_a, "gradient"_a, "batch"_a,
             "gradient <- d(log L)/d(theta) on my columns, from dmb = d(log L)/d(Mb)")
        .def("linear_gm", &KinematicMesh::linear_gm,
             "green"_a, "mb"_a, "prediction"_a, "batch"_a, "residual"_a,
             "prediction <- Gb mb, or Gb mb - prediction when {residual}")
        .def("iterations", &KinematicMesh::iterations, "batch"_a,
             "the most passes of the eikonal over the last {batch} samples");
}

// end of file
