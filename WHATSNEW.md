# What's new in AlTar 2

*September 2026*

AlTar 2 has had a major overhaul. This note summarizes what is new, why it was done, what changes
for existing runs, and how to get started. The full documentation is at
https://altar2.readthedocs.io.

## 1. A new framework

The framework has been rebuilt around three components, all set from a configuration file
(`.pfg`, or now `.yaml`, see section 7):

- the **model**: the forward problem, its data likelihood and the priors of its parameters;
- the **controller**: the sampling algorithm, i.e. the annealing schedule, the sampler that moves
  the chains, and the archiver that saves the results;
- the **job**: the number of chains and steps, and where it runs (cpu or GPU, one process or
  many under MPI, a workstation or a cluster via SLURM).

Under the hood:

- the cpu and GPU versions are now one code base: each component has a `native` and a `cuda`
  implementation, and picks one when the run starts;
- the sampler states and archivers are unified: `altar.bayesian.recorder` keeps results in memory,
  and `altar.bayesian.h5recorder` writes each β step to `step_nnn.h5` and the posterior to
  `step_final.h5`;
- the C/C++ extensions have moved from the raw CPython C-API to pybind11;
- AlTar builds with CMake or with mm, and the GPU parts build only if a CUDA toolkit and a
  CUDA-enabled pyre are found;
- the linear and seismic models (static, static with C_p, kinematic, and joint static + kinematic
  as an ensemble of models) are ported. A new `forward` action runs the model on the posterior
  samples and compares the predictions with the observed data: χ²/N with the full covariance,
  the variance reduction and the data log likelihood, and, given a reference posterior, how the
  two compare. A `resolution` action computes the Fisher information of the data, how well they
  resolve each parameter, and the effective number of parameters they constrain; `synthetic`
  and `recover` run recovery and checkerboard tests; `diagnose` reports the annealing schedule of
  a run and compares its posterior with a reference;
- the volcano models, Mogi, CDM and Reverso, are ported, with the fixes of their open pull
  requests; their GPU forward models are python kernels, written with NVIDIA's cuTile.

## 2. Switching between cpu and GPU

The same configuration runs on either. To use the GPU, set

```none
job.gpus = 1                 ; or --job.gpus=1 on the command line
job.precision = float32      ; optional; float64 is the default, on the cpu and the GPU
```

`job.gpuprecision` still works: if set, it overrides `job.precision` on the GPU.

Several GPUs are used by running several MPI processes, one GPU each (`job.gpuids` picks which
ones). You no longer need separate `altar.cuda.*` components or a separate GPU configuration. The
one exception is the kinematic model, which runs only on the GPU.

## 3. New samplers and controllers

**Why.** CATMIP moves the chains at each β step with a random-walk Metropolis sampler. Its
proposal is a Gaussian shaped by the covariance of the samples, which works well for a modest
number of parameters. As the dimension grows, though, the random walk has to take ever smaller
steps to keep its acceptance rate. The number of steps needed to decorrelate the chains grows
roughly in proportion to the number of parameters. Kinematic inversions and finely discretized
faults, with hundreds to thousands of parameters, are exactly where this becomes the bottleneck.

Gradient-based samplers use the slope of the posterior to make long, directed moves that are
still accepted. In theory, they need far fewer steps as the dimension grows: about d^(1/3) for
MALA and d^(1/4) for HMC, against d for the random walk. The price is one gradient of the data
likelihood per step, which the linear and seismic models now provide.

Two more needs came up often:
- sampling the posterior directly at β = 1, without an annealing ladder, when the prior is
  already informative or a good starting population is available;
- one-line configurations, instead of assembling a controller, a sampler and a scheduler by
  hand.

Before this work, HMC existed only as GPU code that was never connected to a controller. SGLD
ran only on the GPU, and varying the number of Metropolis steps required a separate sampler
class (`AdaptiveMetropolis`).

**What has been implemented.**

| controller | algorithm |
|---|---|
| `altar.bayesian.catmip` | CATMIP: annealing with the COV scheduler, Metropolis sampling |
| `altar.bayesian.mcmc` | Metropolis at β = 1, adapting the proposal between rounds *(new)* |
| `altar.bayesian.catmip_hmc` | CATMIP annealing with Hamiltonian Monte Carlo *(new)* |
| `altar.bayesian.hmc` | HMC at β = 1 *(new)* |
| `altar.bayesian.catmip_mala` | CATMIP annealing with the Metropolis-adjusted Langevin algorithm *(new)* |
| `altar.bayesian.mala` | MALA at β = 1 *(new)* |
| `altar.bayesian.cf_catmip` | cross-fade CATMIP, from the conjugate posterior of a linear model (section 9) *(new)* |
| `altar.bayesian.langevin` | stochastic gradient Langevin dynamics (SGLD), now also on the cpu and under MPI |

- **Samplers are components.** Every annealing controller combines three parts: a sampler
  (`altar.bayesian.samplers.{metropolis,hmc,mala}`), a scheduler (COV or constant β) and an
  archiver. The controllers above are presets of these combinations, and you can still combine
  them yourself.
- **HMC** runs on the cpu, on the GPU and under MPI:
  - each chain follows a leapfrog trajectory of `leapfrog_steps` steps, then the end point is
    accepted or rejected on the change of total energy;
  - the step size adapts after every trajectory rather than once per β step, because CATMIP can
    reach β = 1 in only a handful of β steps;
  - under MPI, the acceptance counts of all processes are summed, so every process uses the same
    step size;
  - a diagonal mass matrix is re-estimated every `mass_update_interval` trajectories from the
    variance of the whole population (across processes), which absorbs the very different
    scales of the parameters. On the linear example, this allowed an equilibrium step size
    about 10 times larger at the same acceptance rate;
  - the likelihoods and gradients of the current chains carry over from one trajectory to the
    next, so the cost is one model evaluation per leapfrog step.
- **MALA** is HMC with a single leapfrog step per proposal, aimed at an acceptance rate of 0.574.
  It costs one gradient per proposal and is a good first choice when gradients are expensive.
- **mcmc** runs Metropolis at β = 1 in `rounds`, `job.steps` steps each, and re-estimates the
  proposal covariance and scaling between rounds, as CATMIP does between β steps.
- **hmc** also walks in `rounds`, with the mass matrix re-estimated between them, and its
  scheduler can replace the outlier chains during a burn-in (`scheduler.burnin`, after ter Braak,
  2006). Chains that start deep in the logit space of a bounded prior can otherwise stay stuck
  there, and shrink everyone's step size. On the Illapel static inversion, `hmc` with 5 rounds of
  100 trajectories and a 3-round burn-in matches a long `catmip_hmc` run in 8 s, against 22 s for
  the shortest `catmip_hmc` that does.
- **SGLD** now runs on the cpu, single-process and MPI, as well as on the GPU.
- **Gradients for the seismic models:** the static model, the moment-magnitude prior and the
  kinematic model have gradients. The kinematic gradient is computed by the chain rule through
  the slip history, on the GPU. An ensemble of models (the joint static + kinematic inversion)
  combines the gradients of its members. The gradient-based samplers therefore work on all of
  the seismic examples.
- **Step sizes and step counts are separate components.** They are shared by all samplers:
  - step size regulators: target acceptance rate (the default), linear in the acceptance rate,
    dual averaging (as in NUTS), and fixed. Each sampler sets its own optimal target: 0.234 for
    Metropolis, 0.7 for HMC, 0.574 for MALA;
  - step counters: a fixed number of steps, or stepping until the chains decorrelate from where
    they started. The decorrelating counter replaces `AdaptiveMetropolis`, and
    `MetropolisVaryingSteps` is gone.
- **Validation.** `models/linear/tests/posterior.py` runs each sampler on the linear example,
  whose posterior is known exactly: a Gaussian, since the data are linear in the parameters. It
  compares the means, standard deviations and correlations of the samples with the exact ones,
  on the cpu or with `--gpu` on the GPU.

## 4. Reparameterization of bounded priors

**Why.** Most geophysical priors are bounded: slip within a range, rake within a sector, a
magnitude window. Metropolis handles bounds easily, by rejecting proposals that fall outside
them. Gradient-based samplers cannot:
- an HMC or MALA trajectory that crosses a bound meets a log density of −∞ and no meaningful
  gradient, so it is wasted;
- SGLD has no accept/reject step at all, so it would simply walk out of the support.

The standard remedy is to sample an unbounded variable and map it into the bounds. Earlier
versions of AlTar did this with separate "logit" twins of each prior (`uniform_logit`,
`tgaussianlogit`, `momentlogit`). These ran on the GPU only, archived values in the transformed
space, and needed an extra `tophysical` pass to convert them back. Every bounded prior needed
its own twin, and results were easy to misread.

**What has been implemented.**
- **One setting on the existing priors.** Setting `reparameterize = True` samples the prior
  through a transform:

  ```none
  prior = uniform
  prior:
      support = (-0.5, 20)
      reparameterize = True
  ```

  The transform is a component (`altar.distributions.transforms`). The logit transform,
  x = a + (b − a) σ(s), is the default and currently the only one, with cpu and GPU
  implementations.
- **Two spaces, kept consistent.**
  - The chains move in the unbounded sampling space s, while the model and the prior always see
    the physical values x.
  - The log-Jacobian log|dx/ds|, which the target density in s includes, is kept in its own
    buffer. As a result, the archived prior and posterior are densities of the physical values.
  - Gradients are carried into sampling space by the chain rule, with a GPU kernel for the
    truncated Gaussian.
  - Resampling in CATMIP reorders both spaces together.
- **Any sampler.**
  - HMC, MALA and SGLD move in sampling space.
  - Metropolis walks the physical values, rejects proposals outside the bounds, and keeps the
    sampling-space values in step, so it gives the same results with or without the setting.
  - The same configuration therefore works with every controller, on the cpu, the GPU and
    under MPI.
- **Direct output.** Each archived step holds `<pset>_physical`, `<pset>_sampling` and the
  log-Jacobian `jacobian`. The `tophysical` action has been retired.
- **Supported priors:** `uniform`, `tgaussian` and the seismic `moment` prior, which constrains
  the moment magnitude Mw. Gradient samplers refuse a bounded prior that is not reparameterized
  and name it, rather than silently producing wrong samples.
- **Validation.** The linear posterior test includes cases with a wide, reparameterized uniform
  prior, checked against the exact posterior.

## 5. Model uncertainty, C_p, for any model

**Why.** The forward model is uncertain too: it rests on an imperfect knowledge of the elastic
structure, the fault geometry, and so on. With large InSAR and GNSS data sets, these epistemic
errors often exceed the observational errors. Ignoring them gives posteriors that are too narrow
and biased toward over-fitting the data. Following Minson et al. (2013) and Duputel et
al. (2014), AlTar accounts for them with a prediction covariance C_p. The likelihood then uses
C_χ = C_d + C_p in place of C_d.

Until now, C_p was only available through a dedicated seismic model,
`altar.models.seismic.cuda.staticcp`. It ran only on the GPU, applied only to the static
inversion, and had its own configuration.

**What has been implemented.**
- **C_p is a component of every model** (`model.cp`), with three policies:
  - `altar.models.cp.none`: no C_p, the default;
  - `altar.models.cp.fixed`: a precomputed C_p, added to C_d once before sampling;
  - `altar.models.cp.adaptive`: C_p re-estimated at each β step from the mean of the current
    samples, which firms up as β increases.
- **How the adaptive update works.** At each β step:
  1. the mean of the current samples is taken over all MPI processes;
  2. the model computes C_p for that mean model, `compute_cp`;
  3. the framework rebuilds and factors C_χ, and the model refreshes whatever depends on it,
     `covariance_updated`, such as the Green's functions merged with the covariance.

  Options:
  - `start` delays C_p until β reaches a given value;
  - `initial_model` and `initial_until` use a given source model to estimate C_p while β is
    small and the mean is still poor.

  Each update is logged with its β and the trace of C_p.
- **Static model.** It computes C_p = K_p C_μ K_p^T, where column i of K_p is K_i θ, from the
  sensitivity kernels K_i (`kmu_file`) and C_μ (`cmu_file`), on both the cpu and the GPU.
- **Works with the gradient samplers.** The gradient of the likelihood uses C_χ, so HMC, MALA
  and SGLD all run with C_p. The example is `staticCp_logit.pfg`.
- **Checked against the data.** The `forward` action estimates C_p at the posterior mean and
  reports the data uncertainty with and without it (`data/sigma` and `data/sigma_chi`).
- **For model developers.** Any model can take part by providing `compute_cp(theta)`. A model
  that cannot is still able to use a fixed C_p.
- **Practical notes.** With a dense C_d, or with any C_p, the covariance is factored as a full
  N_obs × N_obs matrix: once, or at each β step for an adaptive C_p. For many observations, set
  `dataobs.cd_dtype = float64` and keep an eye on memory.
- **Fix to the cpu L2 norm.** With a full (non-diagonal) data covariance, the cpu L2 norm applied
  L instead of Lᵀ of the Cholesky factor of C⁻¹, which gave a wrong likelihood. This is fixed.
  If you ran cpu inversions with a full C_d on earlier versions, those likelihoods were
  affected.

## 6. Built on the latest pyre, with a new CUDA package

AlTar now runs on current pyre (C++23, Python ≥ 3.11), from the `altar2` branch of
github.com/lijun99/pyre. That branch carries pyre changes that are **not yet merged** into
pyre/pyre, the main one being a new `pyre.cuda` package:

- grids on CUDA managed memory, readable from numpy and from CUDA kernels;
- thin cuBLAS, cuSOLVER and cuRAND bindings;
- DLPack and `__cuda_array_interface__` support, so pyre grids can be handed to other GPU
  libraries without a copy.

It uses cuda-python to find the GPUs, so GPU users need `cuda-python` in their environment. We
plan to submit these changes upstream. Until they are merged, please build pyre from the `altar2`
branch.

## 7. YAML configuration files

pyre now reads YAML, so AlTar runs can be configured with a `.yaml` file instead of a `.pfg`
one. Nothing changes in AlTar itself; you only need `pyyaml` in your environment. A `.yaml` file is
found like a `.pfg` one, by the application name (e.g. `linear.yaml`) or with
`--config=run.yaml`. `.pfg` files keep working, and remain the format of the examples.

`models/linear/examples/linear_gpu.yaml` is a complete example, running on the GPU:

```yaml
linear:
  model: altar.models.linear
  controller: altar.bayesian.catmip
  job:
    gpus: 1                # 0 runs on the cpu
    gpuprecision: float64  # or float32
    gpuids: [0]
    chains: 2**10
linear.model:
  case: patch-9
  psets_list: [all]
  psets:
    all: contiguous
linear.model.psets.all:
  count: "{linear.model.parameters}"
  prior: gaussian
  prior.sigma: 0.5
```

Three things differ from `.pfg`:
- **Keys can't repeat** in YAML, so a component chosen in one block (`model: altar.models.linear`)
  is configured in a block of its own (`linear.model:`) or with dotted keys (`prior.sigma: 0.5`).
- **References in braces must be quoted**, as in `"{linear.model.parameters}"`. Unquoted, YAML
  reads them as a mapping and the setting silently gets a wrong value.
- **Keys with a family must be quoted**, e.g. `"mpi.shells.mpirun # altar.plexus.shell":`,
  since an unquoted `#` starts a comment.

`models/linear/tests/config.py` checks that a YAML translation of `linear.pfg` configures the run
exactly as the `.pfg` does.

## 8. numpy on the cpu

**Why.** On the cpu, AlTar kept its samples, densities and data in GSL matrices and vectors
(`altar.matrix`, `altar.vector`), reached through pyre's Python bindings. The arithmetic was
fast, but the code reached it one chain at a time: the likelihood, the priors, the bounds checks
and the Metropolis accept/reject each looped over the chains in Python, and every row access
allocated a new GSL vector. A two-parameter regression took minutes. Writing a model meant
learning the GSL interface (`getRow`, `setRow`, `clone`, BLAS calls with flag enums) rather than
the numpy most of us already use, and the cpu needed a C++ library, `libaltar`, for the COV
solver, the covariance conditioning and the resampling.

**What has been implemented.**
- **numpy arrays throughout the cpu path.** The samples (samples × parameters), the densities
  (samples,), the data and the Green's functions are plain numpy arrays, and every step works on
  all the chains at once.
- **Faster.** The linear posterior tests, 12 cases, run in 31 s on the cpu instead of 19.5
  minutes; the CATMIP regression example in about 2 s.
- **Writing a model.** `forward_model` receives numpy rows, e.g. `prediction[:] = slope * x +
  intercept`, and gradients are numpy expressions; the regression model in `models/regression`
  is the example. `altar.matrix`, `altar.vector`, `altar.blas`, `altar.lapack`, `altar.pdf` and
  `altar.histogram` are gone, with no aliases, and `io.load` returns numpy arrays.
- **Random numbers** come from a numpy generator: `rng.seed` works as before, and
  `rng.algorithm` picks the bit generator (`pcg64` by default). Each MPI process draws from its
  own stream, derived from the seed.
- **The COV scheduler** solves for the next β in Python. Brent's root finding is now the default
  solver: as accurate as the old C++ one, and about twice as fast. The grid search is still
  available (`solver = grid`) and reproduces the old grid solver's β steps exactly. Resampling
  and covariance conditioning are numpy too.
- **Precision.** `job.precision` (`float64` by default, or `float32`) sets the precision on the
  cpu as well as on the GPU. The samples, the data and the forward models take it; the log
  densities, and the sums that make them, stay in double precision. In single precision, the
  cpu runs of models dominated by matrix products, such as those with large Green's functions,
  are faster and use half the memory.
- **No more GSL.** AlTar's core is pure Python: `libaltar` and the `altar` extension module are
  removed, and the C++ forward models of the volcano models take numpy arrays. GSL is no longer
  needed to build or run AlTar.
- **Type annotations.** The framework's methods now declare the types of their arguments and
  results, for editors and type checkers.
- **Validation.** The linear posterior tests pass on the cpu, on the GPU and on 2 MPI processes,
  in both precisions. The regression, volcano and static seismic examples recover their
  posteriors, and the Python COV solvers were replayed against the C++ ones on recorded seismic
  and linear runs.

## 9. Cross-fade CATMIP and the evidence

**Why.** CATMIP anneals from the prior, where the chains know nothing of the data, to the
posterior. For a model that is linear in its parameters with Gaussian errors, the posterior under
a Gaussian prior is known exactly. Cross-fade sampling (Minson, 2024, GJI 239, 1629) starts from
that conjugate posterior instead, fades the actual prior in and the Gaussian one out, and never
evaluates the forward model while doing so. The same weights that drive the annealing also give
the evidence of the model, p(d), which is what model comparison needs.

**What has been implemented.**
- **`altar.bayesian.cf_catmip`**, the cross-fade controller: CATMIP's COV scheduler and Metropolis
  sampler, with the model wrapped in a cross-fade model, `altar.models.crossfade`, which starts the
  chains from the conjugate posterior and fills the two densities of the annealing. The linear and
  the static slip models support it; another linear model does once it provides its conjugate
  posterior (see the Programming Guide). Configurations change by one line,
  `controller = altar.bayesian.cf_catmip`.
- **The evidence** is estimated by the COV scheduler, for CATMIP and cross-fade CATMIP alike,
  printed at the end of a run as `log evidence`, and saved with each step as
  `Annealer/log_evidence`. It assumes that the chains start from the prior (`prep` = `prior`).
- **`softuniform`**, a uniform prior with logistic edges (Minson, 2024, eq. 16): smooth and
  positive everywhere, so that cross-fading needs no special start.
- **Validation.** On the linear example, cross-fade CATMIP reaches the exact posterior in one or two
  β steps, with its evidence within 0.03 nats of the exact value, Gaussian, uniform and soft
  uniform priors alike; CATMIP's evidence comes within 2.5 nats. The 9-patch static example
  takes two β steps instead of about twenty.
- **Where it doesn't pay.** When the bounds of the prior bind, cross-fading loses its edge. The
  Illapel static inversion has about 70 slips at their lower bound: with hard bounds cross-fade
  can't start, and with soft ones it takes 55 s to come within 2% of the posterior spread, where
  `hmc` with a burn-in takes 8 s. It is a tool for data-dominated linear problems.

## What changes for existing runs

- **Configurations need updating.** The old component names are gone, with no aliases:
  - `plainhmc` is now `hmc`, and `catmiphmc` is now `catmip_hmc`;
  - samplers are selected as `altar.bayesian.samplers.<name>`;
  - `altar.cuda.bayesian.metropolis` is replaced by `metropolis` with `job.gpus = 1`;
  - `altar.cuda.data.datal2`, `altar.cuda.models.parameterset` and `altar.cuda.distributions.*`
    are replaced by `datal2`, `contiguous` and the distributions' plain names.

  The example `.pfg` files in `models/*/examples` show the current syntax.
- Reparameterized runs archive physical values directly, so drop any `tophysical` step.
- **Random numbers differ** from earlier versions, since they now come from numpy: a run
  reproduces an older one statistically, not sample for sample.
- **The default COV solver is now Brent's method**, so the β steps differ slightly, within the
  solver's tolerance; `controller.scheduler.solver = grid` gives the old schedule.
- **Models written against `altar.matrix`/`altar.vector`** need porting to numpy, which usually
  makes them shorter; see the Programming Guide.
- Rebuild pyre from `altar2` first, then AlTar. The Installation Guide covers a conda setup and
  both CMake and mm.

## Getting started

- Installation: https://altar2.readthedocs.io/en/latest/guide/Installation.html
- Quick Start: https://altar2.readthedocs.io/en/latest/guide/QuickStart.html
- User Guide (controllers, samplers, priors, C_p, job): https://altar2.readthedocs.io/en/latest/guide/Manual.html
- Programming Guide (writing a model with cpu and GPU implementations): https://altar2.readthedocs.io/en/latest/guide/Programming.html

The documentation of the previous CUDA version stays at https://altar.readthedocs.io.

Please try it on your own problems and report issues at
https://github.com/AlTarFramework/altar/issues, or reach us on the Slack group. Feedback on the
new samplers and on reparameterization in real inversions is especially welcome.
