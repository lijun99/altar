// -*- c++ -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <pybind11/numpy.h>

#include "gaussianslip.h"
#include <altar/models/seismic/cuda/cudaGaussianSlip.h>


namespace {
    using namespace altar::cuda::extensions;
    template <typename io_t, typename real_t>
    using model_t = altar::models::seismic::cudaGaussianSlip<io_t, real_t>;
    using ints_t = py::array_t<int, py::array::c_style | py::array::forcecast>;
    using longs_t = py::array_t<long, py::array::c_style | py::array::forcecast>;
    using doubles_t = py::array_t<double, py::array::c_style | py::array::forcecast>;

    // the device pointer of {g}, of cells {T}
    template <typename T>
    auto cells(grid_t & g, const char * name) -> T *
    {
        const char * format = sizeof(T) == 8 ? "d" : "f";
        if (g.view().format != format) {
            throw py::value_error(std::string("cudaGaussianSlip: '") + name + "' must be a "
                                  + (sizeof(T) == 8 ? "float64" : "float32") + " grid");
        }
        return reinterpret_cast<T *>(g.address());
    }

    // the model of each pair of precisions behind one interface
    struct Engine {
        virtual ~Engine() = default;
        virtual void loglikelihood(grid_t & theta, grid_t & TI, std::size_t stride, grid_t & out,
                                   std::size_t batch) = 0;
        virtual void fractions(grid_t & theta, grid_t & TI, std::size_t stride, grid_t & F, std::size_t batch) = 0;
    };

    template <typename io_t, typename real_t>
    struct Typed : Engine {
        template <typename... Args>
        Typed(Args &&... args) : model(std::forward<Args>(args)...) {}

        void loglikelihood(grid_t & theta, grid_t & TI, std::size_t stride, grid_t & out, std::size_t batch) override
        {
            model.loglikelihood(cells<io_t>(theta, "theta"), theta.shape()[1], cells<io_t>(TI, "TI"), stride,
                                cells<io_t>(out, "out"), batch);
        }

        void fractions(grid_t & theta, grid_t & TI, std::size_t stride, grid_t & F, std::size_t batch) override
        {
            model.fractions(cells<io_t>(theta, "theta"), theta.shape()[1], cells<io_t>(TI, "TI"), stride,
                            cells<real_t>(F, "F"), batch);
        }

        model_t<io_t, real_t> model;
    };

    // the marginal likelihood of kinematicgaussianslip on the gpu, see cudaGaussianSlip.h; theta,
    // the arrival times and the likelihoods in {precision}, the fractions and the linear algebra
    // in {solve}
    class GaussianSlip {
    public:
        GaussianSlip(std::size_t Np, std::size_t Nsp, std::size_t bins, std::size_t components, double dt,
                     int kind, double peak, int mode, longs_t originColumns, double origin,
                     longs_t riseColumns, long scaleColumn, double sigma, doubles_t GG, doubles_t GSD,
                     doubles_t Rinv, doubles_t C0inv, doubles_t Ps, doubles_t bs, ints_t order,
                     std::size_t bounded, doubles_t uniforms, double constant, std::size_t chunk,
                     const std::string & precision, const std::string & solve)
        {
            std::size_t n = 2*Np;
            if (GG.size() != components*n*n || GSD.size() != components*n*bins
                || Rinv.size() != components*bins*bins || C0inv.size() != n*n || Ps.size() != n*n || bs.size() != n
                || order.size() != n
                || riseColumns.size() != Np || (bounded && uniforms.size() % bounded)) {
                throw py::value_error("cudaGaussianSlip: inconsistent sizes");
            }
            std::size_t paths = bounded ? uniforms.size()/bounded : 0;
            auto make = [&](auto * tag) -> std::unique_ptr<Engine> {
                using engine_t = std::remove_pointer_t<decltype(tag)>;
                return std::make_unique<engine_t>(Np, Nsp, bins, components, dt, kind, peak, mode,
                    originColumns.size(), originColumns.data(), origin, riseColumns.data(), scaleColumn,
                    sigma, GG.data(), GSD.data(), Rinv.data(), C0inv.data(), Ps.data(), bs.data(), order.data(),
                    bounded, uniforms.data(), paths, constant, chunk);
            };
            if (precision == "float64" && solve == "float64") {
                _engine = make((Typed<double, double> *) nullptr);
            } else if (precision == "float32" && solve == "float64") {
                _engine = make((Typed<float, double> *) nullptr);
            } else if (precision == "float32" && solve == "float32") {
                _engine = make((Typed<float, float> *) nullptr);
            } else {
                throw py::value_error("cudaGaussianSlip: precision '" + precision + "' with solve '" + solve
                                      + "'; float64 with float64, or float32 with float32 or float64");
            }
        }

        // out <- the marginal log likelihoods of the first {batch} rows of {theta}
        auto loglikelihood(grid_t & theta, grid_t & TI, std::size_t stride, grid_t & out, std::size_t batch) -> void
        {
            _engine->loglikelihood(theta, TI, stride, out, batch);
            cudaCheckError("cudaGaussianSlip.loglikelihood");
            synchronize("cudaGaussianSlip.loglikelihood");
        }

        // F <- the cumulative slip fractions of the first {batch} rows of {theta}
        auto fractions(grid_t & theta, grid_t & TI, std::size_t stride, grid_t & F, std::size_t batch) -> void
        {
            _engine->fractions(theta, TI, stride, F, batch);
            cudaCheckError("cudaGaussianSlip.fractions");
            synchronize("cudaGaussianSlip.fractions");
        }

    private:
        std::unique_ptr<Engine> _engine;
    };
}


auto
altar::models::seismic::extensions::gaussianslip::__init__(py::module & m) -> void
{
    py::class_<GaussianSlip>(m, "cudaGaussianSlip",
                             "the marginal likelihood of the rupture fronts, the gaussian slips integrated out")
        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, double, int, double, int,
                      longs_t, double, longs_t, long, double, doubles_t, doubles_t, doubles_t,
                      doubles_t, doubles_t, doubles_t, ints_t, std::size_t, doubles_t, double, std::size_t,
                      const std::string &, const std::string &>(),
             "Np"_a, "Nsp"_a, "bins"_a, "components"_a, "dt"_a, "kind"_a, "peak"_a, "mode"_a,
             "origin_columns"_a, "origin"_a, "rise_columns"_a, "scale_column"_a, "sigma"_a, "GG"_a,
             "GSD"_a, "Rinv"_a, "C0inv"_a, "Ps"_a, "bs"_a, "order"_a, "bounded"_a, "uniforms"_a, "constant"_a, "chunk"_a,
             "precision"_a = "float64", "solve"_a = "float64")
        .def("loglikelihood", &GaussianSlip::loglikelihood, "theta"_a, "TI"_a, "stride"_a, "out"_a, "batch"_a,
             "out <- the marginal log likelihoods of the first {batch} rows of {theta}, from the arrival "
             "times at the source points of each front in {TI}, {stride} apart")
        .def("fractions", &GaussianSlip::fractions, "theta"_a, "TI"_a, "stride"_a, "F"_a, "batch"_a,
             "F <- the cumulative slip fractions, (batch, patches, bins), in the solve precision");
}

// end of file
