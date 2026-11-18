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

    // how a time interval gets the slip of a point: the rate at its start, times its length; the
    // slip released in it, exactly; or the change of the average of the cumulative slip over it
    // from the interval before, for data averaged over the intervals
    enum class Integration : int { sample = 0, step = 1, bin = 2 };

    // the cumulative slip V of a point {s} after its onset, a fraction of its slip, and W, the
    // integral of V, with their derivatives with respect to {Tr}; dV/ds = v, dW/ds = V
    template <typename TYPE>
    ALTAR_SEISMIC_HD inline void
    stf_cumulative(TYPE s, TYPE Tr, int kind, TYPE peak, TYPE & V, TYPE & V_Tr, TYPE & W, TYPE & W_Tr)
    {
        V = V_Tr = W = W_Tr = 0;
        if (s <= 0) return;
        if (kind == static_cast<int>(STF::cosine)) {
            if (s >= Tr) {
                V = 1;
                W = s - Tr / 2;
                W_Tr = TYPE(-0.5);
                return;
            }
            const TYPE x = TYPE(2 * M_PI) * s / Tr;
            const TYPE c = cos(x), sn = sin(x);
            const TYPE k = TYPE(1 / (4 * M_PI * M_PI));
            V = s / Tr - sn / TYPE(2 * M_PI);
            V_Tr = s * (c - 1) / (Tr * Tr);
            W = s * s / (2 * Tr) + Tr * (c - 1) * k;
            W_Tr = -s * s / (2 * Tr * Tr) + (c - 1) * k + s * sn / (TYPE(2 * M_PI) * Tr);
            return;
        }
        if (s < peak * Tr) {
            V = s * s / (peak * Tr * Tr);
            V_Tr = -2 * V / Tr;
            W = s * V / 3;
            W_Tr = -2 * W / Tr;
            return;
        }
        // the falling side, and past the rise time, where r = 0
        const TYPE q = 1 - peak;
        const TYPE r = s < Tr ? Tr - s : TYPE(0);
        V = 1 - r * r / (q * Tr * Tr);
        V_Tr = -2 * s * r / (q * Tr * Tr * Tr);
        W = s - (1 + peak) * Tr / 3 + r * r * r / (3 * q * Tr * Tr);
        W_Tr = -(1 + peak) / 3 + r * r / (q * Tr * Tr) - 2 * r * r * r / (3 * q * Tr * Tr * Tr);
    }

    // the fraction {f} of the slip of a point that goes to the time interval [s, s + dt) after its
    // onset, by {mode}, see {Integration}, and its derivatives with respect to {s} and {Tr}
    template <typename TYPE>
    ALTAR_SEISMIC_HD inline void
    stf_interval(TYPE s, TYPE dt, TYPE Tr, int kind, TYPE peak, int mode, TYPE & f, TYPE & df_ds,
                 TYPE & df_dTr)
    {
        if (mode == static_cast<int>(Integration::sample)) {
            TYPE v, dv_ds, dv_dTr;
            stf(s, Tr, kind, peak, v, dv_ds, dv_dTr);
            f = v * dt;
            df_ds = dv_ds * dt;
            df_dTr = dv_dTr * dt;
            return;
        }
        f = df_ds = df_dTr = 0;
        TYPE V0, V0_Tr, W0, W0_Tr, V1, V1_Tr, W1, W1_Tr;
        if (mode == static_cast<int>(Integration::step)) {
            if (s + dt <= 0 || s >= Tr) return;
            TYPE v0, v1, a, b;
            stf_cumulative(s, Tr, kind, peak, V0, V0_Tr, W0, W0_Tr);
            stf_cumulative(s + dt, Tr, kind, peak, V1, V1_Tr, W1, W1_Tr);
            stf(s, Tr, kind, peak, v0, a, b);
            stf(s + dt, Tr, kind, peak, v1, a, b);
            f = V1 - V0;
            df_ds = v1 - v0;
            df_dTr = V1_Tr - V0_Tr;
            return;
        }
        // bin: the averages over [s - dt, s) and [s, s + dt) differ only if the slip grows in them
        if (s + dt <= 0 || s - dt >= Tr) return;
        TYPE V2, V2_Tr, W2, W2_Tr;
        stf_cumulative(s - dt, Tr, kind, peak, V0, V0_Tr, W0, W0_Tr);
        stf_cumulative(s, Tr, kind, peak, V1, V1_Tr, W1, W1_Tr);
        stf_cumulative(s + dt, Tr, kind, peak, V2, V2_Tr, W2, W2_Tr);
        f = ((W2 - W1) - (W1 - W0)) / dt;
        df_ds = ((V2 - V1) - (V1 - V0)) / dt;
        df_dTr = ((W2_Tr - W1_Tr) - (W1_Tr - W0_Tr)) / dt;
    }

} // of namespace altar::models::seismic

// end of file
