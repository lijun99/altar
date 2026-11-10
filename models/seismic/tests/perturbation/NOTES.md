# Perturbation sampling on the static model

The `perturbation` controller and sampler, checked on the 9-patch static example (18 parameters,
108 observations, 1024 chains, cpu).

## Bounds that barely bind: `static_perturbation.pfg`

Strike slips gaussian(0, 0.5), dip slips uniform on (-0.5, 20), no moment constraint. The box holds
all but exp(-2e-5) of the untruncated posterior; 20 trajectories bounce 1.5e-4 times each.

Against `catmip` (100 Metropolis steps) on the same prior (`compare.py`):

| catmip run | mean shift, max / rms (sd) | sd ratio | correlation | log evidence |
|---|---|---|---|---|
| prep uniform on the support, seed 1 | 0.069 / 0.044 | 0.97-1.04 | 0.11 | 404.32 |
| prep uniform on the support, seed 2 | 0.108 / 0.050 | 0.96-1.06 | 0.11 | 404.57 |
| the moment prep of `static.pfg` | 0.089 / 0.048 | 0.94-1.05 | 0.11 | 418.34 |

The perturbation log evidence is 404.476; it varies by 5e-12 over the samples. The moments
agree within the noise of two sets of 1024 samples. CATMIP's evidence is right only when its chains
start from the prior: the moment prep, a gaussian in Mw spread by a dirichlet, puts it 14 nats too
high. The guide already says to start from the prior for the evidence.

## Bounds that bind: `bounded.py`

Dip slips uniform on (0.95, 20), the lower bounds near the posterior means (0.98) of all nine. The
dip slips are anticorrelated (down to -0.81), so only exp(-6.31) = 0.18% of the untruncated
posterior lies in the box. Reference: rejection sampling of 10 M untruncated draws (18229 kept),
and the exact evidence from the closed form times that mass.

| run | mean shift (sd) | sd ratio | correlation | log evidence (exact 398.829) |
|---|---|---|---|---|
| perturbation, 20 trajectories | 0.059 | 0.96-1.02 | 0.081 | 398.870 |
| catmip, 100 steps | 0.073 | 0.97-1.04 | 0.067 | 398.204 |

About 10 bounces per trajectory; GHK gives the box mass as exp(-6.267 ± 0.048) against the
rejection count's exp(-6.307).

From the uniform prior on (0.95, 20), the trajectories reach the posterior quickly (4 M reference
draws):

| trajectories | mean shift (sd) | sd ratio | correlation |
|---|---|---|---|
| 1 | 0.154 | 0.90-1.09 | 0.132 |
| 2 | 0.052 | 0.96-1.04 | 0.085 |
| 5 | 0.069 | 0.94-1.05 | 0.083 |

## Linear harness (`models/linear/tests/posterior.py`)

The perturbation cases pass on the cpu and the gpu (float64 and float32): the evidence matches
the exact one to 0.01 nat for the gaussian, wide uniform and tight uniform (box mass exp(-1.24))
priors, reparameterized or not; `catmip` with the perturbation sampler gets within 0.6 nat.

## Illapel static on cairo, gaussian priors (`illapel/`)

342 parameters (170 strike, 170 dip slips, 2 ramp terms), 1248 observations, C = static.Cxi
(C_d + C_p, fixed). Priors: strike N(0, 1), dip N(0, 5), ramp N(0, 5); the dip slips start from the
moment prep (Mw 8.3) unless noted. 4096 chains, float32, one RTX PRO 6000 (gpu 1), one run at a
time; the wall time includes the startup, about 1 s. Reference: the exact posterior and evidence
(3662.87), in float64 (`illapel/compare.py`). 4096 independent draws give a max mean shift of about
0.05 sd and sd ratios within about 0.97-1.04.

| run | wall (s) | mean shift, max / rms (sd) | sd ratio | correlation | log evidence |
|---|---|---|---|---|---|
| perturbation | 1.5 | 0.049 / 0.016 | 0.968-1.036 | 0.064 | 3662.88 |
| cf_catmip, 100 steps | 1.3 | 0.055 / 0.015 | 0.962-1.036 | 0.073 | 3662.87 |
| cf_catmip, 1000 steps | 1.7 | 0.046 / 0.014 | 0.959-1.033 | 0.070 | 3662.87 |
| hmc 5 × 100, burn-in 3 | 6.6 | 0.526 / 0.096 | 0.963-1.029 | 0.062 | |
| hmc 10 × 100, burn-in 5 | 12.5 | 0.094 / 0.026 | 0.964-1.025 | 0.069 | |
| hmc 20 × 100, burn-in 5 | 22.6 | 0.054 / 0.015 | 0.973-1.030 | 0.063 | |
| hmc 40 × 100, burn-in 5 | 44.6 | 0.047 / 0.015 | 0.970-1.027 | 0.064 | |
| catmip_hmc, 20 trajectories | 16.4 | 0.365 / 0.096 | 0.915-1.131 | 0.226 | 3667.96 |
| catmip_hmc, 50 | 38.5 | 0.091 / 0.028 | 0.966-1.044 | 0.094 | 3668.46 |
| catmip_hmc, 200 | 144.9 | 0.050 / 0.016 | 0.960-1.027 | 0.066 | 3668.23 |
| catmip_hmc, 50, dip slips from the prior | 39.9 | 0.077 / 0.025 | 0.973-1.028 | 0.071 | 3663.09 |
| hmc 20 × 100, dip slips from the prior | 22.6 | 0.046 / 0.015 | 0.967-1.030 | 0.069 | |

With gaussian priors cf_catmip's conjugate prior is the prior itself, so it starts from the exact
posterior and takes a single β step; its Metropolis walk never evaluates the data likelihood, so
1000 steps cost 0.4 s more than 100. It is as fast as perturbation here, but only because the
problem is the conjugate one. hmc needs about 20 rounds of 100 trajectories (15x the perturbation
wall time), catmip_hmc about 200 trajectories per step (100x). catmip_hmc's evidence is 5.4 nats
high when the dip slips start from the moment prep, and right (within 0.2) from the prior: the same
bias as on 9patch.
