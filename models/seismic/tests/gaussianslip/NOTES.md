# kinematicgaussianslip: synthetic

`synthetic.py`: the planar fault of the tests (24 patches), one front (rise 3, velocity 5 km/day,
hypocenter (10, 15) km, origin 1 day), slips drawn from their prior N(0, 0.7^2), data of
3 components x 15 stations x 12 bins of 2 days with the separable noise of `../datakron.py`;
signal rms 3.82, noise rms 2.77.

2026-11-17, cairo cpu, catmip, 1024 chains, 30 steps (`~/scripts/altar/kinematic/gaussianslip.sh`):
194 s, 10 beta steps, log evidence -1026.94.

| parameter | truth | posterior | z |
|---|---|---|---|
| rise time | 3.000 | 3.142 ± 0.765 | 0.19 |
| velocity | 5.000 | 5.200 ± 0.564 | 0.36 |
| hypocenter u | 10.000 | 8.964 ± 1.561 | -0.66 |
| hypocenter v | 15.000 | 15.151 ± 0.995 | 0.15 |
| origin | 1.000 | 0.892 ± 0.449 | -0.24 |

Slips, one draw for each of 256 samples: correlation of the mean with the truth 0.965, rms z 0.80,
73 % within 1 sd, 100 % within 2 sd. A laptop smoke run (128 chains, 5 steps) gave the same picture
(log evidence -1028.37).

## Against kinematicmesh (`compare.py`), 2026-11-18, cairo

The same synthetic for both: 24 patches with a rise time each (truth 3), one velocity (5), the
hypocenter (10, 15), origin 0, slips from N(0, 0.7^2), 3 x 15 x 12 data with separable noise; bin
integration in both; kinematicmesh with the step-function big G of the same static green's
functions (its data of the truth equal gs's to 5.7e-16), the slips sampled under the same prior.

First run, gs with its origin pinned by a uniform prior of width 0.002: 99.9 % of catmip's
proposals invalid, a posterior too narrow and off (velocity 8.8 ± 1.1), log evidence -1046.6;
neither target 0.5 (-1043.8) nor 200 steps (-1040.8) fixed it. Scored by gs's exact marginal
likelihood, the kinematicmesh samples had a median of -1013.0, the truth -1013.1, gs's -1028.7.
Fix: the origin parameter set is optional, the origin fixed by a property instead.

| | truth | gs, cpu | gs, 200 steps | km, catmip, gpu | km, catmip_hmc, gpu |
|---|---|---|---|---|---|
| velocity | 5.00 | 6.58 ± 1.09 | 6.61 ± 1.18 | 6.55 ± 1.08 | 6.60 ± 1.17 |
| hypocenter u | 10.0 | 7.12 ± 3.49 | 7.18 ± 3.73 | 7.50 ± 3.49 | 7.34 ± 3.65 |
| hypocenter v | 15.0 | 16.93 ± 3.32 | 17.04 ± 3.51 | 16.66 ± 3.09 | 16.94 ± 3.39 |
| mean rise | 3.00 | 4.30 ± 0.53 | 4.33 ± 0.54 | 4.28 ± 0.53 | 4.34 ± 0.52 |
| log evidence | | -1033.28 | -1033.16 | -1031.43 | -1033.13 |
| chains x steps | | 2048 x 50 | 2048 x 200 | 4096 x 100 | 1024 x 20 |
| time | | 114 s | 413 s | 81 s | 232 s |

Slips against gs: median |mean difference| / sd 0.05 (km), 0.03 (kmhmc); sd ratio 0.97, 1.00;
rise times 0.08 and 0.04, sd ratios 0.97 and 1.00. Slips within 2 sd of the truth 94-96 % in all.
Marginal likelihood at the samples, median: gs -1013.55, km -1012.97, kmhmc -1013.47, truth
-1013.10. gs ran on 16 cpu threads beside the 200-step run; km on one RTX PRO 6000.
