(altar-framework)=
# The AlTar Framework

This page describes the components of an AlTar application other than the model: the controller
that samples the posterior, with its samplers, schedulers and archivers; the job, which sets the
size of the simulation and where it runs; and the check of the results against the data.

(application)=
## The application

An AlTar application is the root component: it assembles the others, and runs them.

`model`
: the model: the forward problem, its data, and the prior distributions of its parameters (see
  {doc}`Models`);

`controller`
: the sampler of the posterior, `altar.bayesian.annealer` by default (see {ref}`below <controllers>`);

`job`
: the size of the simulation, and how it is deployed (see {ref}`below <job>`);

`rng`
: the random number generator;

`monitors`
: event handlers, e.g. `altar.bayesian.profiler`, which times the phases of the simulation.

The application initializes them, the job first and the model last, and then asks the model for
its posterior. Two kinds of applications come with AlTar:

- a plain application, e.g. `altar-linear`, which samples the posterior;
- a *plexus* (`altar.shells.altar`), e.g. `slipmodel.plexus`, which takes an action as its first
  argument:

  ```bash
  slipmodel.plexus sample --config=static.pfg    # sample the posterior
  slipmodel.plexus forward --config=static.pfg   # check the posterior against the data
  slipmodel.plexus about version                 # the version; `about` alone lists the rest
  ```

Pyre also deploys the application to where it runs, e.g. under MPI or a batch scheduler (see
{ref}`Job <job>`).

(controllers)=
## Controllers

The controller samples the posterior. The choice of controller is the choice of algorithm:

| controller | algorithm |
|---|---|
| `altar.bayesian.catmip` | {ref}`CATMIP <catmip>`: annealing with the COV scheduler, Metropolis sampling |
| `altar.bayesian.mcmc` | Metropolis sampling at a fixed $\beta = 1$, without annealing |
| `altar.bayesian.catmip_hmc` | CATMIP annealing, with Hamiltonian Monte Carlo sampling |
| `altar.bayesian.hmc` | Hamiltonian Monte Carlo at a fixed $\beta = 1$, without annealing |
| `altar.bayesian.catmip_mala` | CATMIP annealing, with Metropolis-adjusted Langevin sampling |
| `altar.bayesian.mala` | the Metropolis-adjusted Langevin algorithm at a fixed $\beta = 1$, without annealing |
| `altar.bayesian.cf_catmip` | {ref}`cross-fade CATMIP <cross-fade>`: from the conjugate posterior of a linear model to its posterior |
| `altar.bayesian.langevin` | stochastic gradient Langevin dynamics (SGLD) |
| `altar.bayesian.annealer` | the base of the annealing controllers, with the sampler and the scheduler left to configure |

e.g.

```none
linear:
    controller = altar.bayesian.catmip
```

The controllers without annealing start the chains from the prior and sample the posterior
directly. `mcmc` walks the chains `rounds` times (16 by default), `job.steps` Metropolis steps
each, and adapts the proposal, its covariance and its scaling, between walks, as CATMIP does between
$\beta$ steps:

```none
linear:
    controller = altar.bayesian.mcmc
    controller.rounds = 16
    job.steps = 2**8 ; Metropolis steps per round
```

`hmc` spends `job.steps` HMC trajectories at $\beta = 1$, and `mala` `job.steps` MALA proposals,
adapting their step size after each.

The annealing controllers are built from these components, each configurable:

`sampler`
: moves the chains at each $\beta$: {ref}`Metropolis <metropolis>`, {ref}`HMC <hmc>` or
  {ref}`MALA <mala>`;

`scheduler`
: chooses the next $\beta$: {ref}`COV or a constant <schedulers>`;

`archiver`
: saves the results: {ref}`in memory or to HDF5 files <archivers>`;

`dispatcher`
: notifies the monitors of the phases of each step.

The controller also runs on a *worker* that matches the job: one process on the cpu, one on a
GPU, or several processes under MPI. The worker is chosen from the job configuration, not
configured itself.

(cross-fade)=
### Cross-fade CATMIP

`altar.bayesian.cf_catmip` implements cross-fade sampling
([Minson, 2024](https://doi.org/10.1093/gji/ggae353)), for models with a conjugate prior
$P_c(\boldsymbol\theta)$, one whose posterior $P_c(\boldsymbol\theta|\mathbf d)$ is known in
closed form, e.g. a Gaussian prior for a linear model with Gaussian errors, whose posterior is
Gaussian. Instead of annealing from the prior to the posterior, it anneals from the conjugate
posterior, fading the prior in and the conjugate prior out,

$$
P_m(\boldsymbol\theta|\mathbf d) \propto P_c(\boldsymbol\theta|\mathbf d)
\left[\frac{P(\boldsymbol\theta)}{P_c(\boldsymbol\theta)}\right]^{\beta_m},
$$

which is the posterior at $\beta = 1$. The chains start from the models that already fit the
data, so it needs far fewer $\beta$ steps, often a single one, and it never evaluates the data
likelihood: the cost of each sample doesn't grow with the number of observations. It is otherwise
CATMIP, with its COV scheduler and Metropolis sampler.

The conjugate prior doesn't change the result, only the number of steps; it is a normal
distribution matched to the mean and the variance of the prior of each parameter set. The linear
model and the static slip model provide their conjugate posterior. The data covariance must stay
fixed during the run: no `adaptive` $C_p$.

With a bounded prior, e.g. a uniform one, the initial samples are drawn from the conjugate
posterior within the support of the prior, the limit of the annealed distribution as
$\beta \to 0$. When too little of the conjugate posterior lies within the support, i.e. when the
data pull the parameters far outside it, the run stops with a message; use `catmip` then. The
{ref}`soft uniform <softuniform>` distribution is positive everywhere, and needs no such start.

```none
linear:
    controller = altar.bayesian.cf_catmip
```

The controller wraps the model in a cross-fade model, `altar.models.crossfade`, which fills the
two densities and draws the initial samples, and leaves everything else, e.g. the parameter sets,
to the model. The cross-fade model can also be configured directly, with the model nested in it,
still under `cf_catmip`:

```none
linear:
    model = altar.models.crossfade
    model:
        model = altar.models.linear
        model:
            case = patch-9
            ...
    controller = altar.bayesian.cf_catmip
```

(samplers)=
## Samplers

At each $\beta$, a sampler updates the chains so that they follow
$P_m(\boldsymbol\theta|\mathbf d) \propto P(\boldsymbol\theta)\, P(\mathbf d|\boldsymbol\theta)^{\beta_m}$;
at $\beta = 1$, that is the posterior. It is chosen in the controller, by its name in
`altar.bayesian.samplers`, which keeps it apart from the controller of the same name, e.g. the
sampler `altar.bayesian.samplers.hmc` and the controller `altar.bayesian.hmc`:

```none
linear:
    controller = altar.bayesian.catmip
    controller:
        sampler = altar.bayesian.samplers.metropolis
        sampler:
            scaling = 0.2
```

(metropolis)=
### Metropolis

`altar.bayesian.samplers.metropolis`, the default, proposes a new sample for each chain from a Gaussian
centered on the current one,

$$
\boldsymbol\theta' = \boldsymbol\theta + \alpha\, \boldsymbol\delta, \qquad
\boldsymbol\delta \sim N(0, \boldsymbol\Sigma),
$$

where $\boldsymbol\Sigma$ is the weighted covariance of the samples, and $\alpha$ a scaling
factor; it accepts or rejects the proposals with the Metropolis–Hastings rule. Proposals that fall
outside the support of a bounded prior are rejected as *invalid*; a reparameterized prior avoids
them, with the walk in its sampling space (see {ref}`reparameterization`). After each $\beta$ step,
the acceptance rate adjusts $\alpha$.

`scaling`
: the initial scaling factor $\alpha$; default 0.1.

`proposal`
: the proposal, `altar.bayesian.gaussianproposal`, with
  - `check_positive_definiteness` (default `True`) and `min_eigenvalue_ratio` (default 0.001):
    condition $\boldsymbol\Sigma$, raising its eigenvalues to at least this fraction of the
    largest;
  - `update_interval` (default 1): recompute $\boldsymbol\Sigma$ every this many $\beta$ steps;
  - `archive_sigma` (default `True`): save $\boldsymbol\Sigma$ with the results.

`stepsizer`
: adjusts $\alpha$ from the acceptance rate (see {ref}`step sizes <stepsizers>`); by default
  `targetedrate`, aiming at an acceptance rate of 0.234.

`stepcounter`
: decides how many Metropolis steps each chain takes at each $\beta$ (see
  {ref}`step counts <stepcounters>`); by default `job.steps`.

(hmc)=
### Hamiltonian Monte Carlo

`altar.bayesian.samplers.hmc` moves each chain along a trajectory of the Hamiltonian dynamics on the
potential $U = -\log P_m(\boldsymbol\theta|\mathbf d)$, integrated with leapfrog steps, and accepts
or rejects its end point. It runs `job.steps` trajectories at each $\beta$. It needs the gradient of
the model's data likelihood, and priors defined on the whole real line: reparameterize bounded
priors (see {doc}`Priors`).

`leapfrog_steps`
: the leapfrog steps of each trajectory; default 10.

`step_size`
: the initial leapfrog step size; default 0.01.

`stepsizer`
: adjusts the step size from the acceptance rate; by default `targetedrate`, aiming at 0.7.

`adapt_mass_matrix`
: adapt a diagonal mass matrix to the variance of the samples; default `True`.

`mass_update_interval`
: update the mass matrix every this many trajectories; default 20.

`min_variance`, `max_variance`
: bounds on the variances the mass matrix adapts to; defaults $10^{-8}$ and $10^{8}$.

```none
linear:
    controller = altar.bayesian.catmip_hmc
    controller:
        sampler:
            leapfrog_steps = 10
            step_size = 0.01
    job.steps = 20 ; trajectories per β step
```

(mala)=
### Metropolis-adjusted Langevin

`altar.bayesian.samplers.mala` proposes, for each chain, a step along the gradient of the log posterior
plus a Gaussian perturbation,

$$
\boldsymbol\theta' = \boldsymbol\theta + \frac{\epsilon^2}{2} \mathbf M^{-1} \nabla \log P_m(\boldsymbol\theta|\mathbf d)
+ \epsilon\, \mathbf M^{-1/2} \mathbf z, \qquad \mathbf z \sim N(0, \mathbf I),
$$

and accepts or rejects it with the Metropolis–Hastings rule. That is HMC with a single leapfrog
step, and it takes the same settings, with `leapfrog_steps` = 1 and a step size aimed at an
acceptance rate of 0.574; it runs `job.steps` proposals at each $\beta$. Like HMC, it needs the
gradient of the data likelihood, and unbounded or reparameterized priors. Each proposal costs one
gradient evaluation, against `leapfrog_steps` for an HMC trajectory, but moves the chains less
far. The `mala` controller samples with it at $\beta = 1$, and `catmip_mala` annealed:

```none
linear:
    controller = altar.bayesian.mala
    job.steps = 2000 ; proposals at β = 1
```

(stepsizers)=
### Step sizes

A step size regulator adjusts the Metropolis scaling or the HMC and MALA step size from the acceptance
rate. All of them take

`step_size`
: the initial value;

`step_window`
: the number of proposals to collect before each adjustment; 1 adjusts after each $\beta$ step
  (Metropolis), trajectory (HMC) or proposal (MALA); 0 or less keeps the step size fixed;

`min_step_size`, `max_step_size`
: its bounds; defaults $10^{-4}$ and 1.

The regulators are

| regulator | rule |
|---|---|
| `altar.bayesian.targetedrate` | $s \leftarrow s \exp[g (r - r_0)]$, towards a target acceptance rate $r_0$ (`target`, `gain` $g$ = 1); the default |
| `altar.bayesian.linearrate` | $s = a + b\, r$ (`intercept` $a$ = 0.01, `slope` $b$ = 0.05) |
| `altar.bayesian.dual` | dual averaging, as in NUTS, towards `target` (`gamma`, `t0`, `kappa`) |
| `altar.bayesian.fixedstep` | a fixed step size |

where $r$ is the acceptance rate. The target defaults to the sampler's: 0.234 for Metropolis, 0.7
for HMC, 0.574 for MALA.

(stepcounters)=
### Step counts

A step counter decides how many Metropolis steps each chain takes at each $\beta$:

`altar.bayesian.fixedsteps`
: a fixed number, `steps`, by default `job.steps`; the default.

`altar.bayesian.decorrelating`
: steps until the chains have moved away from where they started: in blocks of
  `corr_check_steps` (default 1000) steps, after at least `min_mc_steps` (default 1000), until the
  correlation between the current and the starting samples falls below `target_correlation`
  (default 0.6), or `max_mc_steps` (default 10000) is reached. With `beta_stage2`, a different
  maximum, `max_mc_steps_stage2`, applies once $\beta$ exceeds it.

```none
linear:
    controller:
        sampler:
            stepcounter = altar.bayesian.decorrelating
            stepcounter:
                min_mc_steps = 3000
                max_mc_steps = 10000
                target_correlation = 0.6
```

(schedulers)=
## Schedulers

A scheduler chooses the next $\beta$, and resamples the chains.

(cov-scheduler)=
### COV

`altar.bayesian.cov`, the scheduler of CATMIP, chooses $\beta_{m+1}$ by the coefficient of
variation of the importance weights of the samples,

$$
w(\boldsymbol\theta_k) = \frac{P_{m+1}(\boldsymbol\theta_k|\mathbf d)}{P_m(\boldsymbol\theta_k|\mathbf d)}
= P(\mathbf d|\boldsymbol\theta_k)^{\beta_{m+1} - \beta_m},
$$

which sets the effective sample size of the resampling,
$\mathrm{ESS} = N_s / (1 + \mathrm{COV}(w)^2)$, with $\mathrm{COV}(w) = \sigma_w / \bar w$. A COV
of 1 keeps half of the samples effective. The samples are then resampled by their weights.

The weights also give the evidence of the model, $\log P(\mathbf d) \approx \sum_m \log \bar w_m$
([Ching and Chen, 2007](https://doi.org/10.1061/(ASCE)0733-9399(2007)133:7(816))), with
$\log P_c(\mathbf d)$ added for cross-fade CATMIP. It is printed at the end of the run, as
`log evidence`, and saved with each step, as `Annealer/log_evidence`. It is an estimate: with a
few hundred chains, it is typically within a nat or two, and on the low side. It assumes that the
initial samples come from the prior, i.e. that the `prep` of each parameter set is its prior, and,
for cross-fade CATMIP, that the priors are normalized: the magnitude penalty of the `moment` prior,
for one, is not.

`target`
: the COV to aim at; default 1.

`solver`
: the solver for $\beta_{m+1}$: `brent` (Brent's root finding, the default; the COV grows with
  the step, so the target is bracketed) or `grid` (an iterative grid search); each has a
  `tolerance` on the COV, 0.01 by default.

`check_positive_definiteness`, `min_eigenvalue_ratio`
: condition the covariance of the samples; defaults `True` and 0.001.

`beta_min`
: the smallest $\beta$ step to take; default 0.

`beta_resampling_start`
: resample only once $\beta$ reaches this value; default 0.

`use_low_variance_resampler`
: resample with evenly spaced random numbers (systematic resampling); default `False`.

```none
linear:
    controller = altar.bayesian.catmip
    controller:
        scheduler:
            target = 2.0
            solver = grid
```

### Constant

`altar.bayesian.constanttemperature`, the default of the base annealer, keeps $\beta$ at
`beta_start`, 1 by default: the chains sample the posterior directly.

(sgld)=
## Stochastic gradient Langevin dynamics

The `altar.bayesian.langevin` controller moves the chains along the gradient of the posterior with
Gaussian noise, and a step size $\epsilon_t$ that decreases over time, without an accept/reject
step. Like HMC, it needs the gradient of the data likelihood, and unbounded or reparameterized
priors.

`tsteps`
: the number of time steps; default 100.

`sweeps`
: the updates at each time step, with the same $\epsilon_t$; default 1.

`tsteps_report`
: report the statistics every this many time steps.

`scheduler`
: the step size schedule: `altar.bayesian.powerdecay`,
  $\epsilon_t = a / (b + t)^\gamma$ (`a`, `b`, `gamma`, with $\gamma \in (0.5, 1]$), or
  `altar.bayesian.expdecay`, $\epsilon_t = a\, e^{-b t}$ (`a`, `b`).

```none
linear:
    controller = altar.bayesian.langevin
    controller:
        tsteps = 20
        sweeps = 5
        scheduler = altar.bayesian.powerdecay
        scheduler:
            a = 1.0e-4
            b = 1
            gamma = 0.33
```

(archivers)=
## Archivers

The archiver saves the results.

`altar.bayesian.recorder`
: the default: keeps the samples in memory, and writes the history of the annealing,
  $\beta$, the scaling and the acceptance statistics of each step, to
  `output_dir/BetaStatistics.txt`.

`altar.bayesian.h5recorder`
: also writes the samples and their densities to HDF5 files, `output_dir/step_nnn.h5` for the
  $\beta$ steps and `output_dir/step_final.h5` for the posterior (see {doc}`QuickStart` for their
  layout).

Both take

`output_dir`
: the directory for the results; default `results`.

`output_freq`
: save every this many $\beta$ steps; default 1. The final step is always saved.

```none
linear:
    controller:
        archiver = altar.bayesian.h5recorder
        archiver:
            output_dir = results/linear
            output_freq = 3
```

(forward-check)=
## Checking the posterior against the data

The `forward` action of a plexus application runs the model on the posterior samples, and compares
the predicted data with the observed data:

```bash
slipmodel.plexus forward --config=static.pfg
```

It reads the posterior from an archived step, runs the forward model on the posterior mean and on
each sample, and reports

- the RMS residual of the posterior mean model, and its $\chi^2/N$;
- the share of the observations that lie within one and two standard deviations of the
  predicted data, where the standard deviation combines the spread of the predictions over the
  posterior samples with the uncertainty of the data (and that of the model, $C_p$, if the model
  has one; see {doc}`StaticCp`).

For an {doc}`ensemble of models <Kinematic>`, it reports each model separately. Its settings go in
a `forward` section of the configuration:

`theta`
: the samples: an archived step, e.g. `results/step_final.h5`, the default, or a `.txt` or `.h5`
  file with one sample per row.

`dataset`
: the dataset of the samples in a plain `.h5` file; by default the first one.

`samples`
: use at most this many of the samples; by default all of them.

`output`
: the HDF5 file for the results; default `forward.h5`. It holds `theta/mean` and `theta/std`;
  the observed data, their uncertainties with and without $C_p$, and the residual of the mean
  model, in `data/`; and, for each quantity the model predicts, e.g. `data`, its value for the mean
  model and its mean and standard deviation over the samples.

```none
slipmodel:
    forward:
        theta = results/static/step_final.h5
        output = results/static/forward.h5
```

(job)=
## Job

The `job` sets the size of the simulation and how it is deployed:

| setting | default | |
|---|---|---|
| `chains` | 64 | the number of Markov chains per process |
| `steps` | 20 | the Metropolis steps, HMC trajectories or MALA proposals, per $\beta$ step |
| `tasks` | 1 | the number of processes per host |
| `hosts` | 1 | the number of hosts |
| `gpus` | 0 | GPUs per process: 0 for the cpu, 1 for a GPU |
| `precision` | `float64` | `float32` or `float64`, for the computations |
| `gpuprecision` | `precision` | `float32` or `float64`, for GPU computations, if not `precision` |
| `gpuids` | | the GPUs to use on each host |
| `tolerance` | 0.001 | $\beta$ within this of 1 counts as 1 |

### Simulation size

The chains of a process are updated together, as a batch. More chains explore the parameter space
better, but need more memory, cpu or GPU; how much depends on the model and its number of
parameters, so try a few sizes, stopping each run after a $\beta$ step or two. The simulation can
also be spread over several processes, on one or several hosts: the total number of chains is
`hosts * tasks * chains`.

At each $\beta$ step, each chain takes `job.steps` Metropolis steps (or HMC trajectories, or MALA proposals), to
equilibrate from one $\beta$ to the next. CATMIP adapts its $\beta$ steps to the samples, so more
steps help, but aren't required.

### Several processes on one host

Several processes run under MPI: set `tasks` and use the `mpi.shells.mpirun` shell,

```none
linear:
    job:
        tasks = 8
    shell = mpi.shells.mpirun
    ; more options for mpirun, if needed
    shell.extra = -mca btl self,tcp
```

or, on the command line,

```bash
altar-linear --config=linear_catmip.pfg --job.tasks=8 --shell=mpi.shells.mpirun
```

The application launches `mpirun` itself, so don't start it with `mpirun` yourself. Without the MPI
shell, a job with more than one task stops with an error. If there is more than one MPI on the machine, tell pyre
which one to use (see {doc}`Installation`). Choose the number of tasks by the physical cores:
hyperthreads rarely help compute-heavy models.

### Several hosts

Without a batch scheduler, list the hosts in a hostfile, e.g. `my_hostfile`,

```none
# hosts, and the processes each can run
192.168.1.101 slots=16
192.168.1.102 slots=16
```

and pass it to the `mpirun` shell:

```none
linear:
    job:
        hosts = 2
        tasks = 8
    shell = mpi.shells.mpirun
    shell:
        hostfile = my_hostfile
```

With the [Slurm](https://slurm.schedmd.com/documentation.html) batch scheduler, use the
`mpi.shells.slurm` shell:

```none
linear:
    job:
        hosts = 4
        tasks = 8
    shell = mpi.shells.slurm

mpi.shells.slurm:
    submit = True ; submit the job; with False, only write the slurm script
    queue = gpu ; the queue
```

With `submit = False`, edit the generated script as your cluster requires, and submit it with
`sbatch`.

(gpu)=
### GPUs

The GPU support uses [CUDA](https://developer.nvidia.com/cuda-toolkit), so it needs NVIDIA GPUs.
Switch to the GPU with

```none
linear:
    job.gpus = 1
```

or `--job.gpus=1` on the command line. The configuration stays the same otherwise: the same model,
priors and controller run on the GPU. Not every model has both implementations: see the model's
page; the kinematic model, for one, runs only on the GPU.

Each process uses one GPU, so `job.gpus` is 0 or 1, and several GPUs are used by several
processes: e.g. for 8 GPUs on each of 4 hosts,

```none
linear:
    job.hosts = 4
    job.tasks = 8
    job.gpus = 1
    shell = mpi.shells.mpirun
```

The processes of a host use its GPUs in order, 0, 1, 2, ...; to use others, list them in
`job.gpuids`, e.g. `job.gpuids = [2, 3]`, or make only those visible, with
`export CUDA_VISIBLE_DEVICES=2,3`.

`job.precision` chooses single (`float32`) or double (`float64`, the default) precision, on the cpu
and the GPU; `job.gpuprecision`, if set, overrides it on the GPU. Most consumer GPUs have few double
precision units, so single precision is much faster on them, and enough for many models; but not
for all: the cascaded kinematic model, for one, loses its chains in single precision (see
{doc}`Kinematic`). On the cpu, single precision speeds up the models dominated by matrix products,
e.g. those with large Green's functions, and halves their memory. The samples, the data and the
forward models take the precision; the log densities, and the sums that make them, stay in double
precision.

## The model

The model defines the forward problem, and computes the data likelihood from it; see {doc}`Models`
for the models that come with AlTar, {doc}`Priors` for the prior distributions of their parameters,
and the {doc}`Programming` to write your own.
