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
