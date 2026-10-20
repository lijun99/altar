// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#if !defined(altar_models_seismic_KinematicMesh_h)
#define altar_models_seismic_KinematicMesh_h

#include <cstddef>
#include <vector>

namespace altar {
    namespace models {
        namespace seismic {
            template<typename TYPE> class KinematicMesh;
        }
    }
}


// the kinematic model on a triangulated fault on the cpu, as cudaKinematicMesh: the arrival times by
// the fast iterative method on a mesh refined from the {Np} patches, at the {Nsp} source points of
// each patch by barycentric interpolation, the slips of each patch over time, Mb[samples][Nt][2][Np],
// and the adjoint of all of them; the samples in parallel. The parameters of a sample, in {idx_map}
// order, are the strike slips, dip slips, rise times and rupture velocities of the patches, then the
// hypocenter along strike and down dip, in the surface coordinates of the mesh
template <typename TYPE>
class altar::models::seismic::KinematicMesh
{
public:
    /// Mb <- the slips of each patch over time, for each row of {theta}
    void calculateBigM(const TYPE * theta, TYPE * Mb, std::size_t parameters, std::size_t batch) const;
    /// grad <- d(log L)/d(theta) on my columns, from dMb = d(log L)/d(Mb)
    void gradient(const TYPE * theta, const TYPE * dMb, TYPE * grad, std::size_t parameters,
                  std::size_t batch) const;
    /// T <- the arrival times at the vertices, for each row of {theta}
    void arrivals(const TYPE * theta, TYPE * T, std::size_t parameters, std::size_t batch) const;
    /// the most passes of the fast iterative method over the last {batch} samples
    int iterations(std::size_t batch) const;
    /// the vertices of the mesh
    std::size_t vertices() const { return _V; }

    KinematicMesh(std::size_t Np, std::size_t Nt, double dt, const double * t0s,
                  std::size_t V, const double * xyz, std::size_t E, const int * incPtr,
                  const int * incA, const int * incB, const int * incPatch, std::size_t N,
                  const int * nbPtr, const int * nb, std::size_t VP, const int * vpPtr,
                  const int * vpPatch, const double * patchGeometry, double spacing,
                  std::size_t P, std::size_t Nsp, const int * corners, const double * weights,
                  const std::size_t * idx_map);

private:
    void _hypocenter(const TYPE * M, TYPE * H) const;
    int _eikonal(const TYPE * M, const TYPE * H, TYPE * T, std::vector<char> & cur,
                 std::vector<char> & nxt) const;
    void _interpolate(const TYPE * T, TYPE * TI0) const;
    void _cast(const TYPE * M, const TYPE * TI0, TYPE * Mb) const;
    void _castAdjoint(const TYPE * M, const TYPE * TI0, const TYPE * dMb, TYPE * grad, TYPE * dTI0) const;
    void _eikonalAdjoint(const TYPE * M, const TYPE * H, const TYPE * T, TYPE * dT, TYPE * grad,
                         std::vector<int> & order, std::vector<int> & parent,
                         std::vector<TYPE> & weight, std::vector<TYPE> & gs) const;

    std::size_t _Np, _Nt, _Nsp, _P, _V, _NGb;
    TYPE _dt, _spacing;
    int _maxiter;
    std::vector<TYPE> _t0s, _xyz, _patchGeometry, _weights;
    std::vector<int> _incPtr, _incA, _incB, _incPatch, _nbPtr, _nb, _vpPtr, _vpPatch, _corners;
    std::vector<std::size_t> _idx;
    mutable std::vector<int> _iterations; ///< the passes of each sample of the last batch
    const TYPE _radius = 1.0; ///< the straight ray seeding radius around the hypocenter, in spacings
    const TYPE _steepness = 10.0; ///< how fast the seeds rise past the radius
    const TYPE _cap = 1.e6; ///< the largest arrival time
};

#endif

// end of file
