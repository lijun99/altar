// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// externals
#include <cmath>
// declarations
#include "reverso.h"


namespace {
    const double pi = 3.14159265358979323846;

    // the volume change of a chamber of radius {a} per unit overpressure, in units of
    // pi a^3 / G: 1 for a sphere, 8(1-v)/(3 pi) for a sill
    auto gamma(bool sill, double v) -> double
    {
        return sill ? 8 * (1 - v) / (3 * pi) : 1;
    }

    // the surface displacement, radial and up, at distance {r} per unit overpressure of a
    // chamber of radius {a} at depth {H}
    auto response(bool sill, double r, double H, double a, double G, double v,
                  double & ur, double & uz) -> void
    {
        auto R2 = r*r + H*H;
        auto alpha = sill ? 4 * H*H / (pi * R2) : 1;
        auto f = a*a*a * alpha * (1 - v) / (G * R2 * std::sqrt(R2));
        ur = r * f;
        uz = H * f;
    }
}


void
altar::models::reverso::
displacements(const const_matrix_t & theta, const const_matrix_t & stations,
              const std::size_t * layout, const medium_t & medium, std::size_t batch,
              const matrix_t & predicted)
{
    const auto & [G, v, mu, drho, g, shallowSill, deepSill] = medium;
    auto gamma_s = gamma(shallowSill, v);
    auto gamma_d = gamma(deepSill, v);

    for (std::size_t sample = 0; sample < batch; ++sample) {
        // the model
        auto Qin = theta(sample, layout[QIN]);
        auto H_s = theta(sample, layout[H_S]);
        auto H_d = theta(sample, layout[H_D]);
        auto a_s = theta(sample, layout[A_S]);
        auto a_d = theta(sample, layout[A_D]);
        auto a_c = theta(sample, layout[A_C]);

        // the ratio of the chamber volumes and the length of the conduit
        auto k = std::pow(a_d/a_s, 3);
        auto H_c = H_d - H_s;
        auto gamma_r = gamma_s + gamma_d*k;
        // the characteristic time (eq. 10)
        auto tau = 8 * mu * H_c * gamma_s * gamma_d * k * std::pow(a_s, 3)
            / (G * std::pow(a_c, 4) * gamma_r);
        // the amplitude of the transient, from zero initial overpressures
        auto A = gamma_d*k / gamma_r
            * (drho*g*H_c - 8*gamma_s*mu*Qin*H_c / (pi * std::pow(a_c, 4) * gamma_r));

        for (std::size_t obs = 0; obs < stations.rows; ++obs) {
            auto t = stations(obs, T);
            auto x = stations(obs, X);
            auto y = stations(obs, Y);
            // the overpressures
            auto f0 = A * (1 - std::exp(-t/tau));
            auto f1 = G * Qin * t / (pi * std::pow(a_s, 3) * gamma_r);
            auto dP_s = f1 + f0;
            auto dP_d = f1 - f0 * gamma_s / (gamma_d*k);
            // the displacements
            auto r = std::sqrt(x*x + y*y);
            double ur_s, uz_s, ur_d, uz_d;
            response(shallowSill, r, H_s, a_s, G, v, ur_s, uz_s);
            response(deepSill, r, H_d, a_d, G, v, ur_d, uz_d);
            auto ur = ur_s*dP_s + ur_d*dP_d;
            auto uz = uz_s*dP_s + uz_d*dP_d;
            // the radial direction
            auto phi = std::atan2(y, x);
            predicted(sample, 3*obs + 0) = ur * std::cos(phi);
            predicted(sample, 3*obs + 1) = ur * std::sin(phi);
            predicted(sample, 3*obs + 2) = uz;
        }
    }
}


void
altar::models::reverso::
verify(const const_matrix_t & theta, const std::size_t * layout, std::size_t batch,
       const vector_t & mask)
{
    for (std::size_t sample = 0; sample < batch; ++sample) {
        if (theta(sample, layout[H_D]) <= theta(sample, layout[H_S])) {
            mask[sample] = 1;
        }
    }
}

// end of file
