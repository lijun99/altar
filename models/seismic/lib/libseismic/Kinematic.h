// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#if !defined(altar_models_seismic_Kinematic_h)
#define altar_models_seismic_Kinematic_h

#include <cstddef>
#include <vector>

namespace altar {
    namespace models {
        namespace seismic {
            template<typename TYPE> class Kinematic;
        }
    }
}


// the kinematic model of {Nas} x {Ndd} rectangular patches on the cpu, as cudaKinematic: the arrival
// times by fast sweeping from the hypocenter on a mesh of {Nmesh} x {Nmesh} cells a patch, padded by
// a patch on each side, the slips of each patch over time, Mb[samples][Nt][2][Nas][Ndd], and the
// adjoint of both; the samples in parallel. The parameters of a sample, in {idx_map} order, are the
// strike slips, dip slips, rise times and rupture velocities of the patches, then the hypocenter
// along strike and down dip from the center of the first patch
template <typename TYPE>
class altar::models::seismic::Kinematic
{
public:
    /// Mb <- the slips of each patch over time, for each row of {theta}
    void calculateBigM(const TYPE * theta, TYPE * Mb, std::size_t parameters, std::size_t batch) const;
    /// grad <- d(log L)/d(theta) on my columns, from dMb = d(log L)/d(Mb)
    void gradient(const TYPE * theta, const TYPE * dMb, TYPE * grad, std::size_t parameters,
                  std::size_t batch) const;
    /// T <- the arrival times on the padded mesh, for each row of {theta}
    void arrivals(const TYPE * theta, TYPE * T, std::size_t parameters, std::size_t batch) const;
    /// the cells of the padded mesh
    std::size_t cells() const { return _Nasf*_Nddf; }

    Kinematic(std::size_t Nas, std::size_t Ndd, std::size_t Nmesh, double dsp, std::size_t Nt,
              std::size_t Npt, double dt, const double * t0s, const std::size_t * idx_map);

private:
    void _seed(const TYPE * M, TYPE * T) const;
    TYPE _upwind(const TYPE * M, const TYPE * T, int i, int j) const;
    void _sweep(const TYPE * M, TYPE * T, TYPE * snapshots) const;
    void _interpolate(const TYPE * T, TYPE * TI0) const;
    void _cast(const TYPE * M, const TYPE * TI0, TYPE * Mb) const;
    void _castAdjoint(const TYPE * M, const TYPE * TI0, const TYPE * dMb, TYPE * grad, TYPE * dTI0) const;
    void _interpolateAdjoint(const TYPE * dTI0, TYPE * dT) const;
    void _sweepAdjoint(const TYPE * M, const TYPE * snapshots, TYPE * lambda, TYPE * grad) const;

    std::size_t _Nas, _Ndd, _Nmesh, _Nt, _Npt, _Npatch, _Nasf, _Nddf, _NGb;
    TYPE _dsp, _dt, _h;
    std::vector<TYPE> _t0s;
    std::vector<std::size_t> _idx;
    const TYPE _cap = 1.e6; ///< the largest arrival time
    const int _sweeps = 1; ///< the passes of the four sweeps
    const TYPE _radius = 1.0; ///< the straight ray seeding radius around the hypocenter, in mesh cells
    const TYPE _steepness = 10.0; ///< how fast the seeds rise past the radius
};

#endif

// end of file
