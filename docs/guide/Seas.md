(seas-model)=
# SEAS: Sequences of Earthquakes and Aseismic Slip

`altar.models.seas.seas3d` infers the frictional properties of a subduction interface, $\log_{10}
\alpha_h$ on a grid of fault patches, from geodetic time series. Each sample runs a sequence of
earthquake cycles on the GPU until it settles into a steady cycle, the *spin-up*. It then runs a
final period over the observation times and predicts the surface displacements. The physics, the
mesh and the earthquake sequence come from seqeas, set up by
an `.ini` file. AlTar samples the parameters with CATMIP, and runs one forward model per sample on
the GPU. The application is `seas`:

```bash
seas --config=seas.yaml
```

This page describes the model's settings, the spin-up and its tolerances, the warm starts, the
outputs, and how to restart or reuse a run. The Anderson acceleration of the spin-up is only on the
`seas-anderson` branch; everything else is on `seas-numpy` as well.

## A configuration

A typical run, in yaml; the `.pfg` form has the same keys:

```yaml
seas:
  controller: altar.bayesian.catmip
  controller.archiver: altar.bayesian.h5recorder
  controller.archiver.output_dir: results
  controller.archiver.output_freq: 1
  controller.sampler: altar.samplers.metropolis
  controller.sampler.scaling: 0.3
  controller.scheduler.beta_resampling_start: 1.0e-3

  job:
    tasks: 8
    gpus: 1
    chains: 256
    steps: 10
    precision: float64
  model: altar.models.seas.seas3d

seas.model:
  case: .
  config_file: fault_subduction.ini
  forward_ode: tractiondependent
  integrator: dopri5
  estimate_row_indices: [0, 1, 2, 3]
  estimate_column_indices: [3, 4, 5, 6, 7]
  # the spin-up (see below)
  spinup_anderson: 5
  # outputs
  record_posterior: 128
  dataobs:
    observations: 17424
    data_file: timeseries.h5
    cd_std: 0.05
    datafile_dataset: flat
    mask_dataset: mask
  psets_list: [log10_alpha_h_0, log10_alpha_h_1, log10_alpha_h_2, log10_alpha_h_3]
  psets:
    log10_alpha_h_0: contiguous
    # ...

seas.model.psets.log10_alpha_h_0:
  count: 5
  prep: uniform
  prep.support: (5, 10)
  prior: uniform
  prior.support: (5, 10)
  prior.reparameterize: True
```

`prior.reparameterize: True` lets the sampler walk in an unbounded space, so that no proposal falls
outside the prior's support. Without it, more than half of the proposals can be invalid and wasted.

Yaml keys can't repeat: a component chosen in one block, e.g. `controller: altar.bayesian.catmip`,
is configured in a block of its own, and a setting given on the command line, e.g.
`--controller.restart=...`, overrides the file. `.pfg` and `.yaml` files can't be mixed in one
configuration.

## The model's settings

The settings of the `seas.model` block:

`config_file`
: the seqeas `.ini` file: the mesh, the earthquakes, the observation times, the integrator's and the
  spin-up's tolerances.

`forward_ode`
: `ratedependent` (the default) integrates the logarithm of the slip rate; `tractiondependent`
  integrates the elastic traction and needs `rho` in the `.ini`.

`integrator`
: `dopri5` (the default), an explicit Runge-Kutta method, or `radau5`, an implicit one for stiff
  problems. `dopri5` is the faster one unless the problem is stiff.

`estimate_row_indices`, `estimate_column_indices`
: the rows and columns of the seqeas `alpha_h_mat` the parameters estimate; the others keep the
  values in the `.ini`.

`cuda_batch_size`
: the samples a GPU runs at once; the chains of a task by default. Lower it if the GPU runs out of
  memory.

`cuda_threads`
: the threads per block of the integrator; 0, the default, picks them from the number of patches.

`ref_station_indices`, `velocity_reference_index`
: reference stations, and the time step to reference all velocities to; off by default.

`verbose`
: 1 logs the timing and the integrator's statistics of every forward call: steps accepted and
  rejected, the spin-up cycles, the smallest step.

`gradient_step`
: the finite-difference step of the gradient, for gradient-based samplers.

The settings of the spin-up and the outputs, described in the sections below:

| setting | default | what it does |
|---|---|---|
| `spinup_anderson` | 0 | accelerate the spin-up by Anderson mixing of this many of its last cycles, at most 8; 0 is off (`seas-anderson` only) |
| `spinup_anderson_damping` | 1.0 | the damping of each Anderson step: 1 takes it whole, less moves only part of the way (`seas-anderson` only) |
| `warm_start` | true | start each forward call's spin-up from where its GPU slot's last one ended |
| `warm_up` | true | before the first forward call, spin up the mean sample alone, and start every slot from it |
| `spinup_file` | none | a step file of an earlier run whose recorded spin-up state the spin-ups start from |
| `record_mean` | true | for each step file the recorder writes, run the mean model and record it there |
| `record_posterior` | 0 | at the end of the run, run this many samples of the final population and write them to `posterior_runs.h5` |

## The spin-up

### Cycles and the convergence test

Each forward call integrates the earthquake cycle repeatedly, from the cycle's start to its end,
until the state at the end of a cycle matches the state at its start. The test compares the
checked half of the state, the slip rates or the tractions, between consecutive cycles, component
by component:

$$
|y_\text{new} - y_\text{old}| \le \texttt{spinup\_atol} + \texttt{spinup\_rtol}\,\max(|y_\text{new}|, |y_\text{old}|)
$$

The test bounds the change per cycle, not the distance to the steady cycle. When each cycle closes
only a small part of the remaining gap, the remaining error can be many times the last change. So
the tolerance has to be tight. With `spinup_rtol = 1` the test passes after one or two cycles,
whatever the state, which is far from converged.

Which of the two tolerances matters depends on the ODE:

- **rate-dependent:** the state is $\log(v/v_0)$, of order 1. An absolute change in it is a relative
  change of $v$, the same on every patch, so `spinup_atol` sets the accuracy and `spinup_rtol`
  should be 0. With rtol > 0, patches whose slip rate is far from $v_0$ would get a looser test,
  for no physical reason. `spinup_atol = 1e-3` bounds the change of $v$ per cycle to 0.1%.
- **traction-dependent:** the state is the traction $\tau \approx (\rho + \log(v/v_0))\,\alpha_h$,
  about $100\,\alpha_h$, i.e. $10^7$ to $10^{12}$ Pa. `spinup_atol` in Pa hardly matters, and
  `spinup_rtol` sets the accuracy. Since $\delta v / v \approx \delta\tau / \alpha_h$,
  `spinup_rtol = 1e-5` bounds the change of $v$ per cycle to about 0.1%.

`n_cycles_max` in the `.ini` caps the cycles of a forward call. A sample that doesn't converge by
then keeps the state it reached. On `seas-anderson` the integrator also allows $10^7$ steps per
forward call ($10^6$ on `seas-numpy`); a sample that exceeds them counts as failed.

### Spin-up and final events

The spin-up cycles use the earthquake sequence of `delta_tau_ix`, which recovers the plate rate over
a cycle. Only the final period, over the observation times, uses `delta_tau_ix_final`, which leaves
the system out of balance on purpose, as the observed coseismic slip doesn't make up the full plate
convergence. (Until October 2026 an uninitialized flag made all but the first two forward calls use
the final events in the spin-up too.)

### The starting state

A cold spin-up starts from seqeas's `v_init`. In current seqeas versions that is the plate velocity
projected on each patch, which is a much better start for the stiff patches, where a fixed high
$\alpha_h$ makes the slip rate relax very slowly. In a traction-dependent test case at
`spinup_rtol = 1e-5`, the plain spin-up converged in 93 cycles instead of not at all within 300, and
the warm-up with Anderson took 13 cycles. `v_init_file` in the `[simulation]` section of the `.ini`
replaces it with any velocity of shape `[inner patches, 2]` in m/s, saved with `numpy.save`.

### Warm starts and the warm-up

A cold spin-up takes tens to hundreds of cycles. With `warm_start`, each GPU slot starts its next
forward call from the slip rates its last call ended with. The traction-dependent ODE builds each
sample's initial traction from them with its own $\alpha_h$. Since nearby samples have similar
steady cycles, a warm call needs far fewer cycles. With a tight tolerance the result doesn't depend
on where the spin-up started, so the warm start only saves time. With a loose one, a sample's
likelihood also depends on what its slot computed before, so keep the tolerance tight.

With `warm_up`, the first forward call is preceded by the spin-up of the population's mean sample
alone, whose state then starts every slot: one system runs the long cold spin-up instead of the
whole batch. A slot whose sample fails to integrate starts its next call cold again.

For each step file the recorder writes, every `output_freq` beta steps and at the end,
`record_mean` runs the mean model, and its state then also starts every slot's next call.

### Anderson acceleration

`spinup_anderson: m` (on `seas-anderson`) treats one cycle as a map from the state at a cycle's
start to the state at its end, and accelerates its fixed-point iteration. After each cycle, it
extrapolates the start of the next one from the last `m` cycles, by the least-squares combination
of their residuals, weighted as the convergence test. Every cycle, including its earthquakes, is
still integrated in full. The spin-up only counts as converged when a cycle returns to its own
start, so the result is a steady cycle of the model, whatever path led there. The history restarts
if the residual grows tenfold, and a failed least-squares solve takes the plain step.

On two samples of a traction-dependent test case, starting from `v_plate`:

| spinup_rtol | spin-up | cycles, cold | cycles, warm | log-likelihoods |
|---|---|---|---|---|
| 1e-5 | plain | 47-51 | 28-35 | 10799.54, 10794.23 |
| 1e-5 | Anderson, 5 | 15-16 | 12-14 | 10799.55, 10794.24 |
| 1e-6 | plain | 293-341 | 49-60 | 10800.17, 10794.86 |
| 1e-6 | Anderson, 5 | 36-37 | 16 | 10800.22, 10794.93 |
| 1e-6 | Anderson, 3 | 40-42 | 18-19 | 10800.21, 10794.94 |
| 1e-7 | Anderson, 5 | 97-99 | 25-27 | 10800.60, 10795.21 |

In a rate-dependent test case (`spinup_atol = 1e-6`), Anderson converges in 68 cycles cold and 17
warm, and a cold and a warm start agree within 0.01.

### Recommended settings

| | rate-dependent | traction-dependent |
|---|---|---|
| `.ini`: `spinup_atol` | 1e-6 with Anderson, 1e-4 without | 1 (hardly matters) |
| `.ini`: `spinup_rtol` | 0 | 1e-6, or 1e-5 for speed |
| `.ini`: `n_cycles_max` | 300 | 300 |
| `seas.model`: `spinup_anderson` | 5 | 5 |

With Anderson, the traction-dependent ODE at `spinup_rtol = 1e-5` is within about 1 log-likelihood
unit of the steady cycle, and at 1e-6 within about 0.5. A plain spin-up at the same tolerance gives
the same value but needs 3-9 times the cycles.

## Outputs

### The step files

With the h5 recorder, each beta step writes a step file `step_NNN.h5`, and the last one
`step_final.h5`, with the samples, beta, the prior, data and posterior log-likelihoods, and the
statistics of the run so far. With `record_mean`, the SEAS group adds:

`SEAS/theta_mean`
: the mean of the population.

`SEAS/v_t0`
: the mean model's slip rates at the first observation time, `[2, patches]` flattened, in m/s; the
  spin-up state a restart or `spinup_file` starts from.

`SEAS/prediction`, `SEAS/likelihood`
: the mean model's predicted data and data log-likelihood, to compare with the observations.

### The posterior runs

With `record_posterior: N`, the model runs N samples drawn without replacement from the final
population after the annealing ends, so that they keep its distribution. It writes
`posterior_runs.h5` to the output directory:

| dataset | shape | |
|---|---|---|
| `theta` | `[N, parameters]` | the samples |
| `rows` | `[N]` | their rows in the population |
| `likelihood`, `failed` | `[N]` | their data log-likelihoods, and which failed to integrate |
| `prediction` | `[N, observations]` | their predicted data |
| `slip`, `slip_rate` | `[N, t_obs, 2, patches]` | the slip [m] and slip rate [m/s] at the observation times |
| `t_obs` | `[t_obs]` | the observation times |
| `ix_estim_row`, `ix_estim_col`, `psets_list`, `mask` | | to map the parameters and data back |

A run restarted from its `step_final.h5` with `record_posterior` set produces it for a finished run.

## Restarting and reusing runs

**Continuing a run** that stopped, e.g. at a time limit: `controller.restart` loads a step file's
samples, beta and proposal scaling, and the annealing continues with the next beta step. A step file
from the current code also brings its mean model's spin-up state.

```yaml
seas:
  controller.restart: results/step_035.h5
  controller.restart_recompute: True
```

`restart_recompute: True` recomputes the likelihoods of the loaded samples. Use it whenever anything
changed since the step was written: the data, `cd_std`, the `.ini`, the code. `False`, the default,
takes them from the file, which saves one forward pass and is fine for continuing the same run.
When in doubt, use True: it costs about one MC step.

**Starting a new run from an earlier run's spin-up**, e.g. variants of a reference case with other
data or prior bounds: `spinup_file` takes the recorded spin-up state from the earlier run's step
file. The new run starts from the prior at beta = 0 as usual; only its spin-ups start from that
state.

```yaml
seas.model:
  spinup_file: /path/to/reference/results/step_final.h5
```

It needs the same mesh and patch order, and a step file that recorded the state; an unusable file
stops the run. `v_init_file` in the `.ini` does the same for a velocity from any source, e.g. a
seqeas spin-up. A state from an earlier, less well spun-up run can be a worse start than `v_plate`:
one recorded with an older seqeas, which started from a poorer `v_init`, spun up far more slowly
than a fresh `v_plate` start.

**Sampling a variant at beta = 1**: when a new case is close to a finished one, plain MCMC can start
from its posterior instead of annealing from the prior:

```yaml
seas:
  controller: altar.bayesian.mcmc
  controller.restart: /path/to/reference/results/step_final.h5
  controller.restart_recompute: True
```

MCMC samples the new posterior from any start. Discard the first rounds as burn-in, and check that
the means and spreads stop moving. If the posterior moves a lot, MCMC alone can miss parts of it
that annealing from the prior would find.

## Notes

- Under MPI only the first task logs, so the integrator's statistics in the log are those of the
  first GPU.
- `job.precision: float64` is needed for inversions: in float32 the likelihood differences between
  nearby samples are much less accurate than the posterior's resolution.
