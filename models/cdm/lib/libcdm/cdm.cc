// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

// declarations
#include "cdm.h"


void
altar::models::cdm::
displacements(const const_matrix_t & theta, const const_matrix_t & stations,
              const std::size_t * layout, double nu, std::size_t batch,
              const matrix_t & predicted)
{
    for (std::size_t sample = 0; sample < batch; ++sample) {
        // the source of this sample
        double p[PARAMETERS];
        for (int i = 0; i < PARAMETERS; ++i) {
            p[i] = theta(sample, layout[i]);
        }
        auto s = source(p);

        for (std::size_t obs = 0; obs < stations.rows; ++obs) {
            auto u = displacement(s, stations(obs, X),
                                  stations(obs, Y), nu);
            // project along the LOS
            vec3<double> n = { stations(obs, LOS_E),
                               stations(obs, LOS_N),
                               stations(obs, LOS_U) };
            auto uLOS = dot(u, n);
            // shift by the offset of the observation's dataset, if any
            auto offset = stations(obs, OFFSET);
            if (offset >= 0) {
                uLOS -= theta(sample, static_cast<std::size_t>(offset));
            }
            predicted(sample, obs) = uLOS;
        }
    }
}


void
altar::models::cdm::
verify(const const_matrix_t & theta, const std::size_t * layout, std::size_t batch,
       const vector_t & mask)
{
    for (std::size_t sample = 0; sample < batch; ++sample) {
        double p[PARAMETERS];
        for (int i = 0; i < PARAMETERS; ++i) {
            p[i] = theta(sample, layout[i]);
        }
        if (!buried(source(p))) {
            mask[sample] = 1;
        }
    }
}

// end of file
