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
displacements(const gsl_matrix & theta, const gsl_matrix & stations,
              const std::size_t * layout, double nu, std::size_t batch,
              gsl_matrix & predicted)
{
    for (std::size_t sample = 0; sample < batch; ++sample) {
        // the source of this sample
        double p[PARAMETERS];
        for (int i = 0; i < PARAMETERS; ++i) {
            p[i] = gsl_matrix_get(&theta, sample, layout[i]);
        }
        auto s = source(p);

        for (std::size_t obs = 0; obs < stations.size1; ++obs) {
            auto u = displacement(s, gsl_matrix_get(&stations, obs, X),
                                  gsl_matrix_get(&stations, obs, Y), nu);
            // project along the LOS
            vec3<double> n = { gsl_matrix_get(&stations, obs, LOS_E),
                               gsl_matrix_get(&stations, obs, LOS_N),
                               gsl_matrix_get(&stations, obs, LOS_U) };
            auto uLOS = dot(u, n);
            // shift by the offset of the observation's dataset, if any
            auto offset = gsl_matrix_get(&stations, obs, OFFSET);
            if (offset >= 0) {
                uLOS -= gsl_matrix_get(&theta, sample, static_cast<std::size_t>(offset));
            }
            gsl_matrix_set(&predicted, sample, obs, uLOS);
        }
    }
}


void
altar::models::cdm::
verify(const gsl_matrix & theta, const std::size_t * layout, std::size_t batch,
       gsl_vector & mask)
{
    for (std::size_t sample = 0; sample < batch; ++sample) {
        double p[PARAMETERS];
        for (int i = 0; i < PARAMETERS; ++i) {
            p[i] = gsl_matrix_get(&theta, sample, layout[i]);
        }
        if (!buried(source(p))) {
            gsl_vector_set(&mask, sample, 1);
        }
    }
}

// end of file
