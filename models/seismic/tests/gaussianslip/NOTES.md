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

## Positive dip slips (`compare.py setup <folder> positive`), 2026-11-19, cairo

The synthetic above with the dip slips of the truth |N(0, 0.7^2)|; gs with `positive = dip`, its
likelihood times the GHK ratio of the box probabilities (256 sobol paths, or 4096); km and kmhmc
with the dip slips under `tgaussian`, support (0, 100). Same data model to 5.0e-16
(`~/scripts/altar/kinematic/compare_positive.sh`).

| | truth | gs | gs, 4096 paths | km, catmip, gpu | km, catmip_hmc, gpu |
|---|---|---|---|---|---|
| velocity | 5.00 | 6.30 ± 1.01 | 6.23 ± 1.00 | 6.22 ± 0.98 | 6.21 ± 0.99 |
| hypocenter u | 10.0 | 6.82 ± 3.15 | 7.06 ± 3.20 | 7.15 ± 3.11 | 7.01 ± 3.27 |
| hypocenter v | 15.0 | 16.56 ± 3.28 | 16.82 ± 3.20 | 16.45 ± 3.25 | 16.72 ± 3.31 |
| mean rise | 3.00 | 4.39 ± 0.51 | 4.37 ± 0.51 | 4.28 ± 0.48 | 4.34 ± 0.49 |
| log evidence | | -1026.02 | -1025.65 | -1024.28 | -1025.63 |
| samples | | 2048 | 2048 | 4096 | 1024 |
| time | | 501 s | 5508 s | 53 s | 220 s |

Slips (gs drawn by SIR over the GHK paths) against gs: median |mean difference| / sd 0.10 (km),
0.03 (kmhmc), 0.02 (4096 paths); sd ratios 0.97-1.01; all within 2 sd of the truth, correlation
0.97. The evidence of 256 paths is 0.37 nats under that of 4096, the bias of the log of the GHK
average; 4096 paths agree with kmhmc to 0.02. gs and gs-4096 ran on 16 threads each, side by side,
12 beta steps each; km and kmhmc one after the other on one RTX PRO 6000.

## cuda against the cpu (`benchmark.py`), 2026-11-19

The likelihoods of 1024 samples, one front, a rise time per patch, float64; cuda equals the cpu to
4e-16 (unit tests on cairo and on a kamb V100). Times, cuda the best of three:

| case | gaussian, cuda | positive dip, cuda | gaussian, cpu | positive dip, cpu |
|---|---|---|---|---|
| 24 patches, 3 x 15 x 12 | 0.015 s | 0.017 s | 0.36 s | 0.63 s |
| 320 patches, 3 x 108 x 24 | 0.33 s | 0.47 s | 2.7 s | 8.7 s |

cuda on a kamb V100 (the cairo RTX PRO 6000 gave 0.36 and 0.45 s at 320 patches; float64 runs go
to kamb or alexandria). The cpu on cairo: at 24 patches, the code before the threads, OpenBLAS on 64; at 320, 48 threads; the 320-patch cpu numbers are 4 x those of
256 samples (`~/scripts/altar/kinematic/profile/stages.sh`). Before the samples were split over
threads, the cpu ran the batched LAPACK calls with OpenBLAS's 64 threads, beside another user's 16
processes, and the 320-patch case had not finished after 2 h 45 min; at 1 or 16 threads it took
about 1.4 s per 64 samples, the solves with the Cholesky factor then general LU solves.

## Static data (`compare.py setup <folder> static`), 2026-11-19, kamb

The synthetic above, slips from N(0, 0.7^2), plus 60 static data of the same slips with 3 ramps
(truth uniform in (-3, 3)) and a dense covariance. gs: `static_data` with the ramps integrated out
under a flat prior, cpu, 16 threads; kmjoint: an ensemble of the static model (ramps uniform in
(-20, 20)) and kinematicmesh, annealed together, catmip, a V100 in float64
(`~/scripts/altar/kinematic/compare_static.sh`). kmjoint's evidence is shown with its ramp prior's
density, 40^-3, taken out.

| | truth | gs | kmjoint |
|---|---|---|---|
| velocity | 5.00 | 7.31 ± 1.14 | 7.06 ± 1.08 |
| hypocenter u | 10.0 | 5.60 ± 3.04 | 5.90 ± 3.08 |
| hypocenter v | 15.0 | 19.32 ± 3.32 | 18.97 ± 2.94 |
| mean rise | 3.00 | 4.11 ± 0.49 | 4.04 ± 0.50 |
| log evidence | | -1125.02 | -1121.79 |
| chains x steps | | 2048 x 50 | 4096 x 100 |
| time | | 383 s | 91 s |

Slips against gs: median |mean difference| / sd 0.03, sd ratio 0.98; rise times 0.12, 0.97;
correlation with the truth 0.989 in both, 94 % within 2 sd. kmjoint's ramps (-2.66, -2.97, 0.53)
± (0.12, 0.15, 0.12), truth (-2.75, -2.95, 0.52). Marginal likelihood at the samples, median:
gs -1101.82, kmjoint -1101.01. kmjoint's catmip evidence is 3.2 nats above gs's exact one; catmip
on kinematicmesh was above by 1.9 (gaussian) and 1.4 (positive) nats in the runs above, with 3
fewer parameters.
