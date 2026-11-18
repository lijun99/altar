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
