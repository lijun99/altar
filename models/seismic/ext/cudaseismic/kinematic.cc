// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// externals
#include <memory>
// my declarations
#include "kinematic.h"
// the model
#include <altar/models/seismic/cuda/cudaKinematic.h>


namespace {
    using namespace altar::cuda::extensions;
    template <typename T> using model_t = altar::models::seismic::cudaKinematic<T>;

    // the cell type of {g}, 'd' or 'f'
    auto cell(grid_t & g) -> char
    {
        auto format = g.view().format;
        if (format.size() != 1 || (format[0] != 'd' && format[0] != 'f')) {
            throw py::value_error("cudaKinematic: unsupported grid cell type '" + format + "'");
        }
        return format[0];
    }

    // the kinematic model at the precision of its {t0s}; the caller keeps {t0s} and {idx_map}
    // alive, since the model only holds their addresses
    class Kinematic {
    public:
        Kinematic(std::size_t Nas, std::size_t Ndd, std::size_t Nmesh, double dsp,
                  std::size_t Nt, std::size_t Npt, double dt, grid_t & t0s,
                  std::size_t samples, std::size_t parameters, std::size_t observations,
                  grid_t & idx_map) :
            _cell(cell(t0s))
        {
            auto format = idx_map.view().format;
            if (format != "q" && format != "l") {
                throw py::value_error("cudaKinematic: idx_map must be an int64 grid, not '" + format + "'");
            }
            auto idx = reinterpret_cast<const std::size_t *>(idx_map.address());
            if (_cell == 'd') {
                _double = std::make_unique<model_t<double>>(Nas, Ndd, Nmesh, dsp, Nt, Npt, dt,
                    reinterpret_cast<const double *>(t0s.address()), samples, parameters, observations, idx);
            } else {
                _float = std::make_unique<model_t<float>>(Nas, Ndd, Nmesh, dsp, Nt, Npt, dt,
                    reinterpret_cast<const float *>(t0s.address()), samples, parameters, observations, idx);
            }
            if (cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cudaKinematic: cublasCreate failed");
            }
        }

        ~Kinematic() { cublasDestroy(_handle); }

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
            done("cudaKinematic.forward_batched");
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
            done("cudaKinematic.cast_mb");
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
            done("cudaKinematic.linear_gm");
        }

    private:
        template <typename T>
        static auto ptr(grid_t & g) -> T * { return reinterpret_cast<T *>(g.address()); }

        auto check(grid_t & a, grid_t & b, grid_t & c) const -> void
        {
            if (cell(a) != _cell || cell(b) != _cell || cell(c) != _cell) {
                throw py::value_error("cudaKinematic: grid precision differs from the model's");
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
altar::models::seismic::extensions::kinematic::__init__(py::module & m) -> void
{
    py::class_<Kinematic>(m, "cudaKinematic", "the kinematic slip model, fast sweeping plus big-G")
        .def(py::init<std::size_t, std::size_t, std::size_t, double, std::size_t, std::size_t, double,
                      grid_t &, std::size_t, std::size_t, std::size_t, grid_t &>(),
             "Nas"_a, "Ndd"_a, "Nmesh"_a, "dsp"_a, "Nt"_a, "Npt"_a, "dt"_a, "t0s"_a,
             "samples"_a, "parameters"_a, "observations"_a, "idx_map"_a,
             py::keep_alive<1, 9>(), py::keep_alive<1, 13>())
        .def("forward_batched", &Kinematic::forward_batched,
             "theta"_a, "green"_a, "prediction"_a, "batch"_a, "residual"_a,
             "prediction <- Gb Mb(theta), or Gb Mb(theta) - prediction when {residual}")
        .def("cast_mb", &Kinematic::cast_mb, "theta"_a, "mb"_a, "batch"_a,
             "mb <- the slips of each patch over time, from {theta}")
        .def("linear_gm", &Kinematic::linear_gm,
             "green"_a, "mb"_a, "prediction"_a, "batch"_a, "residual"_a,
             "prediction <- Gb mb, or Gb mb - prediction when {residual}");
}

// end of file
