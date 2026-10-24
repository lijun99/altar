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
adapting their step size after each. `hmc` can also walk in `rounds` of `job.steps` trajectories,
re-estimating the mass matrix from the samples between them. Chains that start far from the
others, e.g. deep in the logit space of a bounded prior, can stay stuck there, and slow every
chain down by shrinking the step size; a burn-in replaces them between the first walks (see
{ref}`the constant scheduler <constant-scheduler>`):

```none
linear:
    controller = altar.bayesian.hmc
    controller:
        rounds = 5
        scheduler.burnin = 3 ; replace the outlier chains after the first 3 walks
    job.steps = 100 ; trajectories per round
```

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

(pooling)=
### Pooling and the production walk

On the GPU, each chain can keep several of the states it walks through at each $\beta$, not only
the last one, after Dau and Chopin's waste-free SMC. They make up the population the scheduler,
the resampling and the proposal work on, and the archiver records, without more model evaluations:

`pool`
: the states each chain keeps from a walk; default 1, the last one only, which turns pooling off;

`pool_interval`
: the steps or trajectories between them; default 1; 0 spreads them over the length of the
  previous walk, for walks whose length varies, as with the
  {ref}`decorrelating step counter <stepcounters>`.

The states a chain keeps count as more samples only if they are far enough apart to be nearly
independent; `diagnose` reports how many they are worth (see {ref}`diagnose`). Kept from short
walks, they add little.

The annealing ends at the first walk at $\beta = 1$. Its samples share their ancestry, from the
resampling, and a run of thousands of chains can be worth far fewer independent samples. A
*production walk* continues at $\beta = 1$, without resampling and with the sampler as it is, its
step size and mass matrix no longer adapting, for `production` steps or trajectories; with a pool,
the chains keep their states evenly spread over it, and these are the samples of the final step:

```none
linear:
    controller = altar.bayesian.catmip_hmc
    controller:
        pool = 16
        pool_interval = 0
        production = 400 ; trajectories at beta = 1, a state kept every 25
```

The production walk makes up the samples for the analysis: e.g. 1024 chains keeping 16 states a
decorrelation time apart give about 16000 nearly independent samples. It can't move weight between
modes, which only the annealing does.

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

Cross-fading pays when the data dominate the posterior and the bounds of the prior barely matter:
the 9-patch static example takes two $\beta$ steps instead of about twenty. When the bounds bind,
e.g. the positivity of the slip on many patches of an underdetermined slip inversion, the
conjugate posterior is far from the posterior: the start fails with a hard bound, and with a soft
one the annealing still takes many $\beta$ steps, each a Metropolis walk, which needs thousands of
steps with hundreds of parameters. For such problems `catmip_hmc`, or `hmc` with a burn-in, is
faster: on the Illapel static inversion, 342 parameters with about 70 slips at their lower bound,
`hmc` matched the reference in 8 s and `catmip_hmc` in 22 s, where `cf_catmip`, with soft bounds,
needed 55 s to come within 2% of its spread.

The archive of a cross-fade run keeps the samples as usual, but its `prior` and `likelihood` hold
the two densities of the annealing, $\log P_c(\boldsymbol\theta|\mathbf d)$ and
$\log P(\boldsymbol\theta) - \log P_c(\boldsymbol\theta)$, not the prior and the data
likelihood.

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
: the initial scaling factor $\alpha$; by default $2.38/\sqrt{d}$, $d$ the number of parameters,
  the optimal scaling of a random walk shaped by the covariance of a gaussian target (Roberts,
  Gelman and Gilks, 1997). The scaling adapts once per $\beta$ step, by at most a factor of 2, so a
  start far from it costs several steps whose walks barely move the chains.

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
or rejects its end point. It runs `job.steps` trajectories at each $\beta$, or as many as its
`stepcounter` asks for (see {ref}`step counts <stepcounters>`). It needs the gradient of
the model's data likelihood, and priors defined on the whole real line: reparameterize bounded
priors (see {doc}`Priors`).

`leapfrog_steps`
: the leapfrog steps of each trajectory; default 10.

`trajectory_length`
: the length of each trajectory, step size times leapfrog steps, in posterior standard deviations
  under the mass matrix; the leapfrog steps then follow the step size, up to `max_leapfrog_steps`
  (default 200); 0, the default, keeps `leapfrog_steps`. HMC moves a chain ballistically along a
  trajectory but diffusively from one trajectory to the next, so a few long trajectories decorrelate
  the chains for fewer gradients than many short ones; a length of order 1 suits most problems.

`step_size`
: the initial leapfrog step size; by default $d^{-1/4}$, $d$ the number of parameters, or $d^{-1/3}$
  with a single leapfrog step, as in MALA, how the optimal step scales with the dimension on a
  gaussian target (Beskos et al., 2013). It adapts after every trajectory, so the start matters less
  than for Metropolis; a posterior with strongly correlated parameters settles well below it.

`stepsizer`
: adjusts the step size from the acceptance rate; by default `targetedrate`, aiming at 0.7.

`adapt_mass_matrix`
: adapt a diagonal mass matrix to the variance of the samples; default `True`. Without it, the
  unit mass leaves the step size to the narrowest direction of the posterior: on the Illapel
  kinematic case, 4096 chains collapsed to a tenth of the posterior spread.

`mass_matrix`
: `diagonal`, the default, or `dense`, from the covariance of the samples (GPU only); the dense mass
  needs at least twice as many chains as parameters, else the diagonal one is used. Estimated from
  a population that the walks don't mix, it is biased narrow, even with enough chains: with 20
  trajectories per $\beta$ step, 4096 chains on the 690 kinematic parameters came out at 0.73-0.87
  of the posterior spread. Use it with walks that decorrelate.

`mass_shrinkage`
: the shrinkage of the dense mass's correlations toward the identity, from 0 (none, the default)
  to 1 (a diagonal mass).

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

On a hard posterior, e.g. a kinematic inversion of hundreds of correlated parameters, short walks
don't mix: 20 trajectories of 10 leapfrog steps left the chains correlated 0.8-0.9 with where they
started, and runs scattered by 0.3-0.4 posterior standard deviations from each other. What mixed
best for its cost there: trajectories of a fixed length rather than a fixed number of leapfrog
steps, walks until the median correlation drops below 0.3, and fewer chains:

```none
slipmodel:
    controller = altar.bayesian.catmip_hmc
    controller:
        sampler:
            trajectory_length = 1.0
            stepcounter = altar.bayesian.decorrelating
            stepcounter:
                statistic = median
                target_correlation = 0.3
                min_mc_steps = 2
                corr_check_steps = 2
                max_mc_steps = 40
        pool = 16
        pool_interval = 0
    job.chains = 2**10
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

A step counter decides how many Metropolis steps, or HMC trajectories, each chain takes at each
$\beta$:

`altar.bayesian.fixedsteps`
: a fixed number, `steps`, by default `job.steps`; the default.

`altar.bayesian.decorrelating`
: steps until the chains have moved away from where they started: in blocks of
  `corr_check_steps` (default 1000) steps, after at least `min_mc_steps` (default 1000), until the
  correlation between the current and the starting samples falls below `target_correlation`
  (default 0.6), or `max_mc_steps` (default 10000) is reached. With `beta_stage2`, a different
  maximum, `max_mc_steps_stage2`, applies once $\beta$ exceeds it. These defaults are sized for
  Metropolis; HMC trajectories decorrelate in tens, not thousands, so with HMC set them, as in the
  second example below, or each $\beta$ step runs a thousand trajectories.

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

HMC decorrelates in tens of trajectories rather than thousands of steps, so its counts are smaller:

```none
linear:
    controller = altar.bayesian.catmip_hmc
    controller:
        sampler:
            stepcounter = altar.bayesian.decorrelating
            stepcounter:
                min_mc_steps = 5
                corr_check_steps = 5
                max_mc_steps = 200
                target_correlation = 0.3
```

The correlation is that of the chains with where they started, the largest over the parameters, as
{ref}`diagnose <diagnose>` reports it, or the median, with `statistic = median`, which a few slow
parameters among many don't hold back. It is measured against the spread of the chains, so it
can't tell a population that is too narrow: with fewer chains than parameters, the Metropolis
proposal misses directions, which a warning in the journal points out.

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
for one, is not. The bias is not small: on the Illapel static case, the dip slips started from the
`moment` prep gave an evidence 22-80 nats above the one started from the prior, the more the
larger the `target`. The posterior doesn't depend on the prep, which the walks forget; for the
evidence, start from the prior.

The evidence matters only to compare models of the same data, e.g. two fault geometries, with or
without $C_p$, or an algorithm against an exact answer; a single inversion's posterior doesn't
need it, and the run doesn't check its assumptions. To compare evidences: start every parameter set
from its prior; keep the likelihood fixed during the run, i.e. a fixed $C_p$, not one updated from
the samples at each $\beta$; use normalized, deliberately chosen priors, since the evidence pays
for the prior volume the data rule out; use thousands of chains and a target of 1 or below; and
repeat the runs, trusting differences well beyond their spread. Cascaded runs, which start from the
samples of the static stage, have no valid evidence. For linear models, `cf_catmip` gives it
almost exactly.

`target`
: the COV to aim at; default 1. A larger target takes fewer, larger $\beta$ steps: each walk then
  has more copies to separate, but the walks of the Illapel static case took about as long at every
  step, so that the cost followed the number of steps, and the posterior was the same for 0.5, 1
  and 2; a target of 2 took 0.7 of the cost of 1. The evidence gets noisier as the target grows,
  the more so with few chains: 128 chains gave a spread of 1.3, 1.8 and 7 nats between runs for
  0.5, 1 and 2. With thousands of chains and walks that mix, 1.5-2 saves time; with few chains, or
  for the evidence, stay at 1 or below.

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

`resampler`
: replaces the weighted samples by equally weighted ones: `altar.bayesian.importanceresampler`, whose
  `method` draws the copies. Every method copies a sample of weight $w$ $Nw$ times on average; they
  differ in how far the counts stray from it:
  - `multinomial`: $N$ independent draws, so a count is binomial and can be anything from 0 to $N$;
  - `stratified`: one draw in each of $N$ equal slices of $[0, 1)$;
  - `systematic`, the default: one draw $u$ and the evenly spaced $(u + k)/N$, so that a count is
    the floor or the ceiling of $Nw$;
  - `residual`: the floor of $Nw$ copies outright, the rest drawn independently.

  The less the counts stray, the fewer samples the resampling loses by chance, and the more the
  population keeps of the information in the weights.

```none
linear:
    controller = altar.bayesian.catmip
    controller:
        scheduler:
            target = 2.0
            solver = grid
```

(constant-scheduler)=
### Constant

`altar.bayesian.constanttemperature`, the default of the base annealer, keeps $\beta$ at
`beta_start`, 1 by default: the chains sample the posterior directly.

`burnin`
: the walks after which the outlier chains are replaced, as in DE-MC
  ([ter Braak, 2006](https://doi.org/10.1007/s11222-006-8769-1)); 0, none, by default. A chain is
  an outlier when its log posterior is more than `outlier_iqr` interquartile ranges below the
  first quartile of the chains'; it becomes a copy of a randomly chosen other chain. After the
  burn-in the chains are left alone, so the walks that follow sample the posterior.

`outlier_iqr`
: 2 by default.

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

(restart)=
### Restarting a run

An annealing run can continue from any step the HDF5 recorder saved, e.g. after it was stopped or
ran out of time, by naming the step in the controller's `restart`:

```none
linear:
    controller:
        restart = results/linear/step_015.h5
        archiver.output_dir = results/linear_restart
```

The run starts from that step's samples, $\beta$ and data likelihoods, and from the proposal scaling
it reached; it recomputes the prior, and the sampling space of reparameterized priors, so these may
change between the runs. It goes on with the next $\beta$ step, numbering its steps after the
earlier ones, and its `BetaStatistics.txt` covers both runs. The model, its parameter sets and
`cd_std` must be the same, since the data likelihoods are taken from the file. The number of chains
may differ: a population of another size is drawn from the step's, with replacement. Writing to a
fresh `output_dir` keeps the earlier run's files as they were.

Each step file records the statistics of the run so far. Step files of older versions don't; their
history and scaling come from the `BetaStatistics.txt` next to the file, which a run writes when it
finishes, and without one the scaling starts from the sampler's configured value.

The data likelihoods in the file are only right for the model and the data that produced them. After
either changed, e.g. with another version of the model's code, set `controller.restart_recompute =
yes`: the run computes the likelihoods of the step's samples anew, and logs how far they are from the
file's. If they are far apart, every proposal would be judged against the file's values, and few or
none would be accepted.

(forward-check)=
## Checking the posterior against the data

The `forward` action of a plexus application runs the model on the posterior samples, and compares
the predicted data with the observed data:

```bash
slipmodel.plexus forward --config=static.pfg
```

It reads the posterior from an archived step, runs the forward model on the posterior mean and on
each sample, and measures how well they fit the data. With $\mathbf r = \mathbf d - \mathbf p$ the
residual of a prediction $\mathbf p$ of the $N$ observations $\mathbf d$, and $C_\chi$ the
covariance the likelihood uses, $C_d$ or $C_d + C_p$:

- $\chi^2/N = \mathbf r^T C_\chi^{-1} \mathbf r / N$, with the full covariance;
- the variance reduction, $\mathrm{VR} = 1 - \sum r_i^2 / \sum d_i^2$, which ignores the
  covariance: 1 is a perfect fit, 0 no better than predicting zeros, and the large observations
  dominate it;
- the RMS residual, in the units of the data;
- the data log likelihood, $\log L = -\chi^2/2 - \frac12 \log |2\pi C_\chi|$, as the sampler
  sees it.

It reports them for three predictions: of the mean model, the predictions averaged over the
samples (the posterior predictive mean), and the best-fitting sample. Over the samples, it reports
the mean and the standard deviation of $\log L$, and the median of $\chi^2/N$. For a model that
fits within the stated uncertainties, $\chi^2/N$ is about 1 for a posterior sample, and about
$1 - P_{\mathrm{eff}}/N$ for the mean model, $P_{\mathrm{eff}}$ being the number of parameters the
data constrain; much less than 1 means $C_\chi$ is larger than the misfit, much more that the data
are not fit within it. For a posterior close to a gaussian, $2\,\mathrm{var}(\log L) \approx
P_{\mathrm{eff}}$. Finally, it reports the share of the observations that lie within one and two
standard deviations of the predicted data, where the standard deviation combines the spread of the
predictions over the samples with $\sqrt{\operatorname{diag} C_\chi}$ (see {doc}`StaticCp`).

With a `reference`, another posterior of the same model, it measures that one too, with the same
data and $C_\chi$, and reports the difference of their mean $\log L$. Two runs that sample the
same posterior have the same distribution of $\log L$; a run whose samples fit worse and spread
less than another's has likely not converged.

For an {doc}`ensemble of models <Kinematic>`, it reports each model separately. Its settings go in
a `forward` section of the configuration:

`theta`
: the samples: an archived step, e.g. `results/step_final.h5`, the default, or a `.txt` or `.h5`
  file with one sample per row.

`dataset`
: the dataset of the samples in a plain `.h5` file; by default the first one.

`samples`
: use at most this many of the samples; by default all of them.

`reference`
: another posterior to compare with, in the same forms as `theta`; none by default.

`output`
: the HDF5 file for the results; default `forward.h5`. It holds `theta/mean` and `theta/std`;
  the observed data, their uncertainties with and without $C_p$, and the residual of the mean
  model and its whitened form, $L^T \mathbf r$ with $C_\chi^{-1} = L L^T$, in `data/`; for each
  quantity the model predicts, e.g. `data`, its value for the mean model and its mean and standard
  deviation over the samples; and in `fit/`, the measures above, and $\chi^2$ and $\log L$ of each
  sample. The measures of the reference go in `reference/fit/`.

```none
slipmodel:
    forward:
        theta = results/static/step_final.h5
        output = results/static/forward.h5
```

(resolution)=
## How well the data resolve the parameters

The `resolution` action of a plexus application measures how much the data constrain each
parameter, from the Fisher information of the data,

$$
F = J^T C_\chi^{-1} J ,
$$

with $J$ the jacobian of the forward model at the posterior mean, by central finite differences,
all the perturbed parameters run as one batch. With the prior approximated by a gaussian
$N(\mathbf m, C_m)$, matched to the mean and the variance of each parameter set's prior, as in
{ref}`cross-fade CATMIP <cross-fade>`, the linearized posterior covariance is
$(F + C_m^{-1})^{-1}$, and the resolution matrix

$$
R = (F + C_m^{-1})^{-1} F
$$

tells how a change of each parameter would be recovered: its diagonal is near 1 for a parameter
the data resolve, and near 0 for one only the prior constrains. Its trace, $P_{\mathrm{eff}}$, is
the effective number of parameters the data constrain.

```bash
slipmodel.plexus resolution --config=static.pfg
```

For each parameter set, it reports the mean and the range of the diagonal of $R$, its share of
$P_{\mathrm{eff}}$, and the median ratio of the sampled standard deviations to the linearized ones,
$\sqrt{\operatorname{diag}(F + C_m^{-1})^{-1}}$. That ratio is near 1 for a linear model with a
gaussian prior; it departs from 1 where the prior is not gaussian, e.g. bounds cut into the
posterior, where the model is nonlinear over the posterior, or where the run has not converged:
a sampler that has not mixed tends to give too narrow a posterior. For an ensemble of models, $F$
is the sum of each model's, on its own parameters and data. The resolution is local, at the
posterior mean, and ignores the bounds of the priors; see the {ref}`forward check <forward-check>`
for the fit. Its settings go in a `resolution` section:

`theta`
: the posterior: an archived step, e.g. `results/step_final.h5`, the default, or a `.txt` or
  `.h5` file with one sample per row; its standard deviations set the finite-difference steps.

`dataset`
: the dataset of the samples in a plain `.h5` file; by default the first one.

`samples`
: average $F$ over this many posterior samples instead, for nonlinear models; each takes two
  forward runs per parameter.

`step`
: the finite-difference step of each parameter, relative to its posterior standard deviation;
  default 0.001.

`modes`
: the number of the best-resolved patterns to save; default 10.

`output`
: the HDF5 file for the results; default `resolution.h5`. It holds $F$ (`fisher`), the prior
  variances, the diagonal of $R$ (`resolution`), $P_{\mathrm{eff}}$ (`effective_parameters`),
  the linearized standard deviations, the eigenvalues of the prior-whitened
  $C_m^{1/2} F C_m^{1/2}$, each above 1 a direction the data constrain more than the prior, and
  the leading `patterns`, those directions in parameter space, normalized.

(recovery-tests)=
## Recovery and checkerboard tests

A recovery test samples synthetic data made from a known, true model, and checks whether the
posterior finds it: whether the truth lies within its uncertainties as often as it should, and
which features come back. Unlike the {ref}`resolution <resolution>`, it includes the noise, the
priors, their bounds, the nonlinearity of the model and the sampler itself. A checkerboard test
makes the true model, over some parameter sets laid out on a grid, e.g. the slips of a fault,
alternate between two values in squares of a given size; where the squares come back, the data
resolve features of that size. Repeat with a few sizes. A recovered checkerboard doesn't prove
the resolution at every scale: small squares may come back where larger features don't, and the
other way around (Lévêque, Rivera and Wittlinger, *On the use of the checker-board test to assess
the resolution of tomographic inversions*, Geophysical Journal International, 115, 313, 1993).

The `synthetic` action makes the data:

```bash
slipmodel.plexus synthetic --config=static.pfg
```

It takes the true model from the mean of a posterior, replaces the chosen parameter sets by a
checkerboard, runs the forward model on it, adds noise drawn from $C_d$, and writes a copy of the
input directory of the model, with these data in place of the observations, and the true model
in `truth.txt`. A run on the copy, e.g. with `--slipmodel.model.case=synthetic`, samples the
synthetic data with everything else unchanged; for an ensemble of models, each model's data are
replaced. The `recover` action then compares the posterior with the truth:

```bash
slipmodel --config=static.pfg --slipmodel.model.case=synthetic \
          --controller.archiver.output_dir=results/synthetic
slipmodel.plexus recover --config=static.pfg --slipmodel.recover.theta=results/synthetic/step_final.h5
```

For each parameter set, it reports the root mean square and the largest of
$z = (\theta_{\mathrm{true}} - \bar\theta)/\sigma_\theta$, the shares of the parameters whose true
value lies within the central 68% and 95% intervals of the samples, and the correlation of the
posterior mean with the truth. A calibrated posterior has an RMS $z$ near 1 and holds the truth
about 68% and 95% of the time; a posterior that is too narrow, e.g. from a run that has not
converged, has larger $z$ and lower shares.

The settings of `synthetic`:

`theta`
: the true model is the mean of these samples: an archived step, by default
  `results/step_final.h5`, or a `.txt` or `.h5` file with one sample per row.

`dataset`
: the dataset of the samples in a plain `.h5` file; by default the first one.

`checkerboard`
: the parameter sets to replace by a checkerboard; none by default.

`grid`
: the rows and the columns of their parameters, e.g. `[along strike, down dip]` for the slips
  of a fault, the columns running fastest; by default a single column.

`block`
: the size of the squares, in parameters; default 1.

`values`
: the two values of the checkerboard; default `[0, 1]`.

`noise`
: whether to add noise; default `True`.

`cp`
: draw the noise from $C_\chi = C_d + C_p$ at the true model instead, for a model with a $C_p$;
  default `False`.

`seed`
: the seed of the noise; default 0.

`output`
: the directory for the copy of the inputs; default `synthetic`.

The settings of `recover`: `theta` and `dataset`, the posterior, as above; `truth`, the true
model, by default `synthetic/truth.txt`; and `output`, the HDF5 file for the truth, the posterior
mean and standard deviation, $z$ and whether the truth lies within each interval, for each
parameter, by default `recover.h5`.

(diagnose)=
## Checking how a run went

The fit to the data doesn't tell whether the sampler drew from the posterior: a run stuck around
a good model fits well, with too narrow a posterior. The `diagnose` action reads the steps a run
archived, with `altar.bayesian.h5recorder`, and reports its annealing schedule, step by step:

```bash
slipmodel.plexus diagnose --config=static.pfg
```

- $\beta$;
- the acceptance rate of the moves of the step, and the scaling of the proposal, or the step
  size; a rate far from the sampler's target, by default 0.234 for Metropolis, 0.574 for MALA and
  0.7 for HMC, means the proposal didn't adapt;
- the coefficient of variation of the importance weights that led to the step, and their
  effective sample size as a share of the chains, $\mathrm{ESS}/N = 1/(1 + \mathrm{cov}^2)$; COV
  keeps the coefficient of variation at its target, 1, so $\mathrm{ESS}/N \approx 0.5$, but for the
  last step, which reaches $\beta = 1$;
- the running estimate of the evidence, $\log p(\mathbf d)$, from schedulers that estimate it;
- how well the walk of the step mixed, from runs that archive it (`Diagnostics` in the step
  files): `parents`, the distinct samples its resampling kept; `lineages`, the initial samples the
  population descends from; `corr` and `max`, the median and largest, over the parameters, of the
  correlation across the chains between the end of each chain's walk and its start; and, with a
  {ref}`pool <pooling>`, `states`, the effective states per chain, the variance of the population
  over that of the means of the chains.

A walk that leaves the correlation near 1 hasn't moved the chains away from the copies the
resampling made; their spread then shrinks step by step, and the posterior comes out too narrow.
More steps per $\beta$, a better proposal or HMC lower it; the
{ref}`decorrelating step counter <stepcounters>` walks until it is below a target. Few lineages are
expected after many steps and harmless when the walks decorrelate. The effective states per chain
tell how much a pool adds: about 1 when the states a chain keeps are as good as one, up to `pool`
when they are independent; it reads low when sibling chains, copies of one parent, haven't
separated.

Given a `reference` posterior, e.g. from a longer run, with more chains or steps, or from another
sampler, it also reports, for each parameter set, how far the means are from the reference's,
in standard deviations of the reference, and the ratio of the standard deviations. Two runs that
sample the same posterior agree within the Monte Carlo error, e.g. a few hundredths of a standard
deviation with thousands of chains; a ratio below 1 means the run is too narrow, the usual sign of
a run that has not converged.

Its settings, in a `diagnose` section: `theta`, the final step of the run, by default
`results/step_final.h5`, whose directory holds the other archived steps; `reference`, an archived
step, or a `.txt` or `.h5` file with one sample per row, and `dataset`, its dataset in a plain
`.h5` file; and `output`, the HDF5 file for the schedule and the comparison, by default
`diagnose.h5`.

(optimize)=
## The best fit

The `optimize` action finds the best fit of the model to its data, the maximum of the data log
likelihood over the parameters inside the supports of their priors, or, with `posterior`, the
maximum of the posterior under the conjugate prior, the gaussian matched to each parameter set's
prior, as in the {ref}`resolution <resolution>`:

```bash
slipmodel.plexus optimize --config=static.pfg
```

It is the quick answer to how well a model can fit the data at all, without an annealing schedule
or the mixing of walks to worry about: e.g. whether a change to the model, more free parameters
or another fixed input, improves the best fit, or whether a posterior sits well below the best fit,
a sign that its walks did not mix. Its best point is also a starting point for other runs, and a
`theta` for the other actions.

It has three methods, set by `method`:

`cma`, the default
: the covariance matrix adaptation evolution strategy, CMA-ES (Hansen, *The CMA evolution
  strategy: a tutorial*, 2016), which needs no gradient: each generation samples a population from
  a gaussian, runs the forward model on it as one batch, as a run's chains, and moves the gaussian
  toward the best of them, learning the scales and the correlations of the parameters. Points
  outside the bounds are moved to the nearest bound and penalized, so that optima on a bound are
  found. It is the robust choice far from the optimum, or with several of them, for up to a few
  hundred parameters.

`lm`
: Levenberg-Marquardt on the whitened residuals, with Marquardt's scaling of the damping and the
  parameters held on a bound left out of each step; each iteration tries four dampings in one
  batch. It converges in a few iterations near a single optimum, and gives the covariance at the
  optimum, $(J^T J)^{-1}$, the Laplace approximation of the posterior with `posterior`, whose
  standard deviations it reports and saves.

`lbfgs`
: projected L-BFGS, with a backtracking line search whose step lengths are tried in one batch.
  It needs more iterations than `lm` on a least-squares problem, and converges slowly in its
  last digits: a `tolerance` $\epsilon$ on the objective leaves the parameters about
  $\sqrt{2 \epsilon}$ standard deviations from the optimum.

`lm` and `lbfgs` take the jacobian of the residuals by central differences, inside the bounds,
$2n$ forward runs for $n$ parameters in one batch, as the {ref}`resolution <resolution>` does; they
measure their steps in units of the initial standard deviations of `step`, so that parameters of
very different scales are treated alike. A good practice is `cma` first, then `lm` from its best
point, `theta = optimize.txt`, for the polish and the covariance. On the linear example, all
three find the exact posterior mean, `cma` to $10^{-4}$ of a posterior standard deviation in about
500 generations of 12 points for 18 parameters, `lm` to $10^{-9}$ in 5 iterations, with the exact
standard deviations, and `lbfgs` to about $10^{-3}$ in 70 iterations.

Each batch of points is shared among the processes of an mpi run: the first process runs the
optimizer and sends the points, each process runs the forward model on its share, on its own gpu,
and the first collects the results. A model whose forward runs are expensive, e.g. an earthquake
cycle model on several gpus, runs the same way as its sampling, e.g.

```bash
slipmodel.plexus optimize --config=static.pfg --shell=mpi.shells.mpirun --job.tasks=4 --job.gpus=1
```

with the plexus as a script, so that `mpirun` can start it in each process. For an ensemble of
models, the log likelihood is the sum of each model's, on its own parameters; a $C_p$ is computed
once, at the start. Its settings, in an `optimize` section:

`method`
: `cma`, the default, `lm` or `lbfgs`.

`theta`
: the start: an archived step, or a `.txt` or `.h5` file with one sample per row, whose mean is
  used; by default the means of the priors.

`dataset`
: the dataset of the samples in a plain `.h5` file; by default the first one.

`posterior`
: maximize the posterior under the conjugate prior instead of the data log likelihood; default
  no.

`step`
: the initial standard deviation of each parameter, relative to the spread of the samples of
  `theta`, or else to the standard deviation of its prior; default 0.3, and at most a quarter of
  a bounded support. It is also the unit of the steps of `lm` and `lbfgs`.

`population`, `generations`, `seed`
: for `cma`: the points of each generation, by default $4 + 3 \ln n$ for $n$ parameters, a
  larger population searching more globally and costing nothing more per generation while it
  fits in the batch the model runs at once, e.g. on a gpu; the most generations, default 2000;
  the seed of the random populations, default 0.

`iterations`, `difference`, `memory`
: for `lm` and `lbfgs`: the most iterations, default 200; the finite-difference step of each
  parameter, relative to its `step`, default $10^{-3}$; for `lbfgs`, the curvature pairs to keep,
  default 10.

`tolerance`
: stop when the best value changes less than this over a window of generations for `cma`, or of
  5 iterations for `lm` and `lbfgs`; `cma` also stops when every step size shrinks below this
  fraction of its start; default $10^{-6}$.

`batch`
: the most points each process runs at once; by default all of its share.

`output`
: the HDF5 file for the results; default `optimize.h5`. It holds the best point (`best/theta`)
  and its value, its data log likelihood, the start, the bounds, the history of the best values,
  for `cma` the final mean and covariance of the search, and for `lm` the covariance at the
  optimum and its standard deviations (`best/sd`). The best point is also written, as one row, to
  a `.txt` file of the same name, e.g. for the `theta` of the {ref}`forward check <forward-check>`.

The report lists, for each parameter set, the best values and how many of them sit on a bound; a
best fit that needs many parameters on their bounds deserves a look at the bounds, or at the model.

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
