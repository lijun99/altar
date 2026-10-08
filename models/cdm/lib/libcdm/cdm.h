// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// code guard
#pragma once

// the compound dislocation model of Nikkhoo et al. [2017], GJI 208(2), 877-894: three
// mutually orthogonal rectangular dislocations with a common opening, in an elastic half space;
// a transcription of Nikkhoo's CDM.m

#include <cmath>
#include <cstddef>
#include <limits>

namespace altar::models::cdm {

    // a strided view of a matrix, e.g. of a numpy array, in cells
    template <typename cell_t>
    struct matrix_view_t {
        cell_t * data;
        std::size_t rows, cols;
        std::ptrdiff_t rowStride, colStride;

        auto operator()(std::size_t row, std::size_t col) const -> cell_t & {
            return data[static_cast<std::ptrdiff_t>(row) * rowStride
                        + static_cast<std::ptrdiff_t>(col) * colStride];
        }
    };

    // and of a vector
    template <typename cell_t>
    struct vector_view_t {
        cell_t * data;
        std::size_t size;
        std::ptrdiff_t stride;

        auto operator[](std::size_t i) const -> cell_t & {
            return data[static_cast<std::ptrdiff_t>(i) * stride];
        }
    };

    using const_matrix_t = matrix_view_t<const double>;
    using matrix_t = matrix_view_t<double>;
    using vector_t = vector_view_t<double>;

    // the columns of a {stations} matrix: location, LOS unit vector (east, north, up), and the
    // column of the observation's dataset offset in {theta}, or -1 for none
    enum station_t { X = 0, Y, LOS_E, LOS_N, LOS_U, OFFSET, STATION_COLUMNS };

    // the columns of the source parameters in {theta}: the centroid, its depth, the opening,
    // the semi-axes and the rotation angles about the x, y and z axes, in degrees
    enum parameter_t { X0 = 0, Y0, DEPTH, OPENING, AX, AY, AZ, OMEGAX, OMEGAY, OMEGAZ, PARAMETERS };

    // a minimal 3-vector
    template <typename T>
    struct vec3 { T x, y, z; };

    template <typename T>
    inline auto operator+(const vec3<T> & a, const vec3<T> & b) -> vec3<T>
    { return { a.x + b.x, a.y + b.y, a.z + b.z }; }

    template <typename T>
    inline auto operator-(const vec3<T> & a, const vec3<T> & b) -> vec3<T>
    { return { a.x - b.x, a.y - b.y, a.z - b.z }; }

    template <typename T>
    inline auto operator*(T s, const vec3<T> & a) -> vec3<T>
    { return { s * a.x, s * a.y, s * a.z }; }

    template <typename T>
    inline auto dot(const vec3<T> & a, const vec3<T> & b) -> T
    { return a.x * b.x + a.y * b.y + a.z * b.z; }

    template <typename T>
    inline auto cross(const vec3<T> & a, const vec3<T> & b) -> vec3<T>
    { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

    template <typename T>
    inline auto norm(const vec3<T> & a) -> T
    { using std::sqrt; return sqrt(dot(a, a)); }

    // the twelve vertices of the three rectangular dislocations, as in CDM.m
    template <typename T>
    struct source_t {
        vec3<T> P[4], Q[4], R[4];
        T ax, ay, az, opening;
    };

    // build a source from its parameters; {ax, ay, az} are semi-axes, angles are in degrees
    template <typename T>
    inline auto
    source(T x0, T y0, T depth, T opening, T ax, T ay, T az, T omegaX, T omegaY, T omegaZ)
        -> source_t<T>
    {
        using std::sin;
        using std::cos;
        const auto deg = T(3.14159265358979323846) / 180;
        auto sx = sin(omegaX*deg), cx = cos(omegaX*deg);
        auto sy = sin(omegaY*deg), cy = cos(omegaY*deg);
        auto sz = sin(omegaZ*deg), cz = cos(omegaZ*deg);
        // the columns of R = Rz Ry Rx
        vec3<T> R0 = { cz*cy, -sz*cy, sy };
        vec3<T> R1 = { cz*sy*sx + sz*cx, -sz*sy*sx + cz*cx, -cy*sx };
        vec3<T> R2 = { -cz*sy*cx + sz*sx, sz*sy*cx + cz*sx, cy*cx };
        // the axes
        ax *= 2;
        ay *= 2;
        az *= 2;
        // the centroid
        vec3<T> P0 = { x0, y0, -depth };

        source_t<T> s;
        s.ax = ax;
        s.ay = ay;
        s.az = az;
        s.opening = opening;

        s.P[0] = P0 + T(.5)*ay*R1 + T(.5)*az*R2;
        s.P[1] = s.P[0] - ay*R1;
        s.P[2] = s.P[1] - az*R2;
        s.P[3] = s.P[0] - az*R2;

        s.Q[0] = P0 - T(.5)*ax*R0 + T(.5)*az*R2;
        s.Q[1] = s.Q[0] + ax*R0;
        s.Q[2] = s.Q[1] - az*R2;
        s.Q[3] = s.Q[0] - az*R2;

        s.R[0] = P0 + T(.5)*ax*R0 + T(.5)*ay*R1;
        s.R[1] = s.R[0] - ax*R0;
        s.R[2] = s.R[1] - ay*R1;
        s.R[3] = s.R[0] - ay*R1;

        return s;
    }

    // whether the whole source lies below the free surface, the half space solution's requirement
    template <typename T>
    inline auto
    buried(const source_t<T> & s) -> bool
    {
        for (int i = 0; i < 4; ++i) {
            if (s.P[i].z > 0 || s.Q[i].z > 0 || s.R[i].z > 0) {
                return false;
            }
        }
        return true;
    }

    // the surface displacements of an angular dislocation, in its own coordinate system
    template <typename T>
    inline auto
    AngDisDispSurf(T y1, T y2, T beta, const vec3<T> & b, T nu, T a) -> vec3<T>
    {
        using std::sin;
        using std::cos;
        using std::tan;
        using std::sqrt;
        using std::atan2;
        using std::log;
        const auto pi = T(3.14159265358979323846);

        auto sinB = sin(beta);
        auto cosB = cos(beta);
        auto cotB = 1 / tan(beta);
        auto z1 = y1*cosB + a*sinB;
        auto z3 = y1*sinB - a*cosB;
        auto r = sqrt(y1*y1 + y2*y2 + a*a);
        auto c = 1 - 2*nu;

        // the Burgers function
        auto Fi = 2*atan2(y2, (r+a)/tan(beta/2) - y1);

        auto v1b1 = b.x/2/pi*((1-c*cotB*cotB)*Fi + y2/(r+a)*(c*(cotB+y1/2/(r+a))-y1/r)
                              - y2*(r*sinB-y1)*cosB/r/(r-z3));
        auto v2b1 = b.x/2/pi*(c*((T(.5)+cotB*cotB)*log(r+a)-cotB/sinB*log(r-z3))
                              - 1/(r+a)*(c*(y1*cotB-a/2-y2*y2/2/(r+a))+y2*y2/r)
                              + y2*y2*cosB/r/(r-z3));
        auto v3b1 = b.x/2/pi*(c*Fi*cotB + y2/(r+a)*(2*nu+a/r) - y2*cosB/(r-z3)*(cosB+a/r));

        auto v1b2 = b.y/2/pi*(-c*((T(.5)-cotB*cotB)*log(r+a) + cotB*cotB*cosB*log(r-z3))
                              - 1/(r+a)*(c*(y1*cotB+a/2+y1*y1/2/(r+a)) - y1*y1/r)
                              + z1*(r*sinB-y1)/r/(r-z3));
        auto v2b2 = b.y/2/pi*((1+c*cotB*cotB)*Fi - y2/(r+a)*(c*(cotB+y1/2/(r+a))-y1/r)
                              - y2*z1/r/(r-z3));
        auto v3b2 = b.y/2/pi*(-c*cotB*(log(r+a)-cosB*log(r-z3)) - y1/(r+a)*(2*nu+a/r)
                              + z1/(r-z3)*(cosB+a/r));

        auto v1b3 = b.z/2/pi*(y2*(r*sinB-y1)*sinB/r/(r-z3));
        auto v2b3 = b.z/2/pi*(-y2*y2*sinB/r/(r-z3));
        auto v3b3 = b.z/2/pi*(Fi + y2*(r*cosB+a)*sinB/r/(r-z3));

        return { v1b1 + v1b2 + v1b3, v2b1 + v2b2 + v2b3, v3b1 + v3b2 + v3b3 };
    }

    // the surface displacements at (x, y) of the angular dislocation pair on the side PA-PB of a
    // rectangular dislocation with burgers vector {b}
    template <typename T>
    inline auto
    AngSetupFSC(T x, T y, const vec3<T> & b, const vec3<T> & PA, const vec3<T> & PB, T nu)
        -> vec3<T>
    {
        using std::acos;
        using std::abs;
        const auto pi = T(3.14159265358979323846);
        const auto eps = std::numeric_limits<T>::epsilon();

        auto side = PB - PA;
        auto beta = acos(-side.z / norm(side));
        // a vertical side contributes nothing
        if (abs(beta) < eps || abs(pi - beta) < eps) {
            return { 0, 0, 0 };
        }

        // the angular dislocation coordinate system, as the rows of the transformation
        vec3<T> ey1 = { side.x, side.y, 0 };
        ey1 = (1 / norm(ey1)) * ey1;
        vec3<T> ey3 = { 0, 0, -1 };
        auto ey2 = cross(ey3, ey1);

        // the observation point relative to PA and PB, and the burgers vector, in that system
        vec3<T> rA = { x - PA.x, y - PA.y, -PA.z };
        T y1A = dot(ey1, rA);
        T y2A = dot(ey2, rA);
        T y1B = y1A - dot(ey1, side);
        T y2B = y2A - dot(ey2, side);
        vec3<T> bADCS = { dot(ey1, b), dot(ey2, b), dot(ey3, b) };

        // pick the artefact-free configuration for the points near the free surface
        auto angle = (beta*y1A >= 0) ? beta - pi : beta;
        auto vA = AngDisDispSurf(y1A, y2A, angle, bADCS, nu, -PA.z);
        auto vB = AngDisDispSurf(y1B, y2B, angle, bADCS, nu, -PB.z);
        auto v = vB - vA;

        // back to the earth fixed system
        return v.x*ey1 + v.y*ey2 + v.z*ey3;
    }

    // the surface displacements at (x, y) of the rectangular dislocation P1 P2 P3 P4
    template <typename T>
    inline auto
    RDdispSurf(T x, T y, const vec3<T> * P, T opening, T nu) -> vec3<T>
    {
        auto normal = cross(P[1] - P[0], P[3] - P[0]);
        auto b = (opening / norm(normal)) * normal;

        return AngSetupFSC(x, y, b, P[0], P[1], nu)
            + AngSetupFSC(x, y, b, P[1], P[2], nu)
            + AngSetupFSC(x, y, b, P[2], P[3], nu)
            + AngSetupFSC(x, y, b, P[3], P[0], nu);
    }

    // the surface displacements (east, north, up) at (x, y) of the source {s}
    template <typename T>
    inline auto
    displacement(const source_t<T> & s, T x, T y, T nu) -> vec3<T>
    {
        // a rectangle with a vanishing side contributes nothing, and has no normal
        vec3<T> u = { 0, 0, 0 };
        if (s.ay != 0 && s.az != 0) {
            u = u + RDdispSurf(x, y, s.P, s.opening, nu);
        }
        if (s.ax != 0 && s.az != 0) {
            u = u + RDdispSurf(x, y, s.Q, s.opening, nu);
        }
        if (s.ax != 0 && s.ay != 0) {
            u = u + RDdispSurf(x, y, s.R, s.opening, nu);
        }
        return u;
    }

    // a source from its parameters, laid out as in {parameter_t}
    template <typename T>
    inline auto
    source(const T * p) -> source_t<T>
    {
        return source<T>(p[X0], p[Y0], p[DEPTH], p[OPENING], p[AX], p[AY], p[AZ],
                         p[OMEGAX], p[OMEGAY], p[OMEGAZ]);
    }

    // fill the first {batch} rows of {predicted} (samples x observations) with the LOS
    // displacements, less the dataset offsets, of the sources in {theta}
    void displacements(const const_matrix_t & theta, const const_matrix_t & stations,
                       const std::size_t * layout, double nu, std::size_t batch,
                       const matrix_t & predicted);

    // flag in {mask} the first {batch} samples whose source reaches above the free surface
    void verify(const const_matrix_t & theta, const std::size_t * layout, std::size_t batch,
                const vector_t & mask);

} // of namespace altar::models::cdm

// end of file
