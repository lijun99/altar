// -*- C++ -*-
//
// (c) 2013-present parasim inc
// (c) 2010-present california institute of technology
// all rights reserved
//

#pragma once

#include <cmath>

#ifdef __CUDACC__
#define ALTAR_SEISMIC_HD __host__ __device__
#else
#define ALTAR_SEISMIC_HD
#endif

namespace altar::models::seismic {

    // the source time functions, of unit area, over the rise time Tr after the onset of a point
    enum class STF : int { triangle = 0, cosine = 1 };

    // the slip rate v of a point {s} after its onset, and its derivatives with respect to {s} and
    // {Tr}: a triangle peaking at {peak} Tr, symmetric for 0.5, or a raised cosine
    template <typename TYPE>
    ALTAR_SEISMIC_HD inline void
    stf(TYPE s, TYPE Tr, int kind, TYPE peak, TYPE & v, TYPE & dv_ds, TYPE & dv_dTr)
    {
        v = dv_ds = dv_dTr = 0;
        if (s <= 0 || s >= Tr) return;
        if (kind == static_cast<int>(STF::cosine)) {
            const TYPE w = TYPE(2 * M_PI) / Tr;
            const TYPE c = cos(w * s), sn = sin(w * s);
            v = (1 - c) / Tr;
            dv_ds = w * sn / Tr;
            dv_dTr = -(1 - c) / (Tr * Tr) - w * s * sn / (Tr * Tr);
            return;
        }
        const TYPE up = peak * Tr;
        if (s < up) {
            v = 2 * s / (peak * Tr * Tr);
            dv_ds = 2 / (peak * Tr * Tr);
            dv_dTr = -4 * s / (peak * Tr * Tr * Tr);
        } else {
            const TYPE q = 1 - peak;
            v = 2 * (Tr - s) / (q * Tr * Tr);
            dv_ds = -2 / (q * Tr * Tr);
            dv_dTr = 2 * (2 * s - Tr) / (q * Tr * Tr * Tr);
        }
    }

} // of namespace altar::models::seismic

// end of file
