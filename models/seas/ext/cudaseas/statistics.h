// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// (c) 2022 - 2026 california institute of technology
// all rights reserved

// code guard
#ifndef altar_models_seas_ext_cudaseas_statistics_h
#define altar_models_seas_ext_cudaseas_statistics_h

#include <vector>
#include <dopri5/method.cuh>

namespace altar::cuda::py::seas {

    // the step statistics of the last batch, as a dict of lists with one entry per system
    inline auto statistics_dict(const std::vector<::cuda::ode::dopri5::StepStatistics> & stats)
        -> py::dict
    {
        std::vector<int> accepted, rejected, stiff, cycles;
        std::vector<bool> failed;
        std::vector<double> hmin, hmax;
        for (const auto & s : stats) {
            accepted.push_back(s.accepted);
            rejected.push_back(s.rejected);
            stiff.push_back(s.stiff);
            cycles.push_back(s.cycles);
            failed.push_back(s.failed);
            hmin.push_back(s.hmin);
            hmax.push_back(s.hmax);
        }
        auto d = py::dict();
        d["accepted"] = accepted;
        d["rejected"] = rejected;
        d["stiff"] = stiff;
        d["cycles"] = cycles;
        d["failed"] = failed;
        d["hmin"] = hmin;
        d["hmax"] = hmax;
        return d;
    }

} // namespace

#endif
// end of file
