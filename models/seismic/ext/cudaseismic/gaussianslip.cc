// -*- c++ -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include <memory>
#include <pybind11/numpy.h>

#include "gaussianslip.h"
#include <altar/models/seismic/cuda/cudaGaussianSlip.h>


namespace {
    using namespace altar::cuda::extensions;
    using model_t = altar::models::seismic::cudaGaussianSlip;
    using ints_t = py::array_t<int, py::array::c_style | py::array::forcecast>;
    using longs_t = py::array_t<long, py::array::c_style | py::array::forcecast>;
    using doubles_t = py::array_t<double, py::array::c_style | py::array::forcecast>;

    // the double precision device pointer of {g}
    auto doubles(grid_t & g, const char * name) -> double *
    {
        if (g.view().format != "d") {
            throw py::value_error(std::string("cudaGaussianSlip: '") + name + "' must be a float64 grid");
        }
        return reinterpret_cast<double *>(g.address());
    }

    // the marginal likelihood of kinematicgaussianslip on the gpu, see cudaGaussianSlip.h
    class GaussianSlip {
    public:
        GaussianSlip(std::size_t Np, std::size_t Nsp, std::size_t bins, std::size_t components, double dt,
                     int kind, double peak, int mode, longs_t originColumns, double origin,
                     longs_t riseColumns, long scaleColumn, double sigma, doubles_t GG, doubles_t GSD,
                     doubles_t Rinv, doubles_t C0inv, ints_t order, std::size_t bounded,
                     doubles_t uniforms, double constant, std::size_t chunk)
        {
            std::size_t n = 2*Np;
            if (GG.size() != components*n*n || GSD.size() != components*n*bins
                || Rinv.size() != components*bins*bins || C0inv.size() != n*n || order.size() != n
                || riseColumns.size() != Np || (bounded && uniforms.size() % bounded)) {
                throw py::value_error("cudaGaussianSlip: inconsistent sizes");
            }
            std::size_t paths = bounded ? uniforms.size()/bounded : 0;
            _model = std::make_unique<model_t>(Np, Nsp, bins, components, dt, kind, peak, mode,
                originColumns.size(), originColumns.data(), origin, riseColumns.data(), scaleColumn,
                sigma, GG.data(), GSD.data(), Rinv.data(), C0inv.data(), order.data(), bounded,
                uniforms.data(), paths, constant, chunk);
        }

        // out <- the marginal log likelihoods of the first {batch} rows of {theta}
        auto loglikelihood(grid_t & theta, grid_t & TI, std::size_t stride, grid_t & out, std::size_t batch) -> void
        {
            _model->loglikelihood(doubles(theta, "theta"), theta.shape()[1], doubles(TI, "TI"), stride,
                                  doubles(out, "out"), batch);
            cudaCheckError("cudaGaussianSlip.loglikelihood");
            synchronize("cudaGaussianSlip.loglikelihood");
        }

        // F <- the cumulative slip fractions of the first {batch} rows of {theta}
        auto fractions(grid_t & theta, grid_t & TI, std::size_t stride, grid_t & F, std::size_t batch) -> void
        {
            _model->fractions(doubles(theta, "theta"), theta.shape()[1], doubles(TI, "TI"), stride,
                              doubles(F, "F"), batch);
            cudaCheckError("cudaGaussianSlip.fractions");
            synchronize("cudaGaussianSlip.fractions");
        }

    private:
        std::unique_ptr<model_t> _model;
    };
}


auto
altar::models::seismic::extensions::gaussianslip::__init__(py::module & m) -> void
{
    py::class_<GaussianSlip>(m, "cudaGaussianSlip",
                             "the marginal likelihood of the rupture fronts, the gaussian slips integrated out")
        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, double, int, double, int,
                      longs_t, double, longs_t, long, double, doubles_t, doubles_t, doubles_t,
                      doubles_t, ints_t, std::size_t, doubles_t, double, std::size_t>(),
             "Np"_a, "Nsp"_a, "bins"_a, "components"_a, "dt"_a, "kind"_a, "peak"_a, "mode"_a,
             "origin_columns"_a, "origin"_a, "rise_columns"_a, "scale_column"_a, "sigma"_a, "GG"_a,
             "GSD"_a, "Rinv"_a, "C0inv"_a, "order"_a, "bounded"_a, "uniforms"_a, "constant"_a, "chunk"_a)
        .def("loglikelihood", &GaussianSlip::loglikelihood, "theta"_a, "TI"_a, "stride"_a, "out"_a, "batch"_a,
             "out <- the marginal log likelihoods of the first {batch} rows of {theta}, from the arrival "
             "times at the source points of each front in {TI}, {stride} apart")
        .def("fractions", &GaussianSlip::fractions, "theta"_a, "TI"_a, "stride"_a, "F"_a, "batch"_a,
             "F <- the cumulative slip fractions, (batch, patches, bins)");
}

// end of file
