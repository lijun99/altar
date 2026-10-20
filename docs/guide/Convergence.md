(convergence)=
# Convergence: chains, walks and steps

A run always reaches $\beta = 1$, and its samples always fit the data. Neither tells whether they
sample the posterior. This chapter is about what decides that, how to check a run, and how to
choose its settings. The numbers come from three cases:

- the linear example, patch-9: 18 parameters, with an exact posterior;
- the Illapel static inversion: 342 parameters;
- the Illapel cascaded kinematic inversion of 2022: 690 parameters.

Most settings were run two or three times; treat the numbers as indications, not laws.


## What a run spends its budget on

At each $\beta$ step, CATMIP chooses the next $\beta$ from the coefficient of variation (COV) of
the importance weights, resamples the chains by their weights, and walks them with Markov chain
Monte Carlo. The cost is

$$
\text{model evaluations} \approx N \times S \times K \;(\times\; \text{leapfrog steps, for HMC}),
$$

with $N$ the chains, $S$ the steps (or HMC trajectories) of each walk, and $K$ the $\beta$ steps.

| budget | what it controls | too small |
|---|---|---|
| $N$, the chains (`job.chains`) | the Monte Carlo error, about $\sigma/\sqrt{N_\mathrm{eff}}$; the population statistics the sampler adapts to (the proposal $\Sigma$, the mass matrix, the next $\beta$); how many modes the population can carry | a proposal or mass from too few samples; a schedule that runs ahead; a lost mode |
| $S$, the walk (`job.steps`, or a {ref}`step counter <stepcounters>`) | how far the chains move from the copies the resampling made | the copies stay together: too narrow a posterior, and runs that disagree with each other |
| $K$, the $\beta$ steps | set by the {ref}`COV target <cov-scheduler>` and by how much the data tell | rarely the problem |

The walk and the chains don't substitute for each other: more chains don't fix walks that are too
short, and longer walks don't fix a proposal estimated from too few chains.


## How runs go wrong

**Walks too short.** The resampling copies the likely samples and drops the others, about half of
them at each step. The walk must then separate the copies. If it doesn't, the population shrinks
step after step:

- Illapel static, Metropolis, 4096 chains and 1000 steps per $\beta$: the posterior spreads came out
  at 0.87-0.92 of the reference; 10000 steps were needed.
- Illapel kinematic, HMC with 20 trajectories of 10 leapfrog steps: the chains ended each walk still
  correlated 0.8-0.9 with where they started. Three such runs disagreed with each other by 0.3-0.4
  posterior standard deviations in the slips, and up to 1 in the hypocenter.

**A unit mass for HMC.** Without a mass matrix the leapfrog step is set by the narrowest direction
of the posterior, and the walks barely move in the others. The kinematic case, with 4096 chains and
`adapt_mass_matrix = False`, gave spreads of a tenth of the posterior's. The default, a diagonal
mass, fixes this at no cost.

**Too few chains for Metropolis.** The Metropolis proposal is the covariance of the population. With
fewer chains than parameters it misses directions altogether. 128 chains on the 342 static
parameters gave spreads of 0.43 of the posterior's. The run warns when this happens. HMC with the
same 128 chains was right.

**A dense mass from a population that doesn't mix.** It is biased narrow, even with more than twice
as many chains as parameters: 0.73-0.87 of the spread, on the kinematic case with short walks.

**Rows are not samples.** The rows of a run descend from fewer and fewer ancestors, and the copies
of one parent stay correlated. For the total potency of the kinematic case, a 4096-row run with
short walks was worth about 7 independent samples; one with long walks, about 100.

**The fit to the data doesn't tell.** All these runs fit the data about equally well. A run stuck
around a good model fits well, with too narrow a posterior.


## Checking a run

The {ref}`diagnose <diagnose>` action reads the archived steps and reports, for each $\beta$ step,
how well its walk mixed:

`corr`, `max`
: the median and largest correlation, across the chains, of the end of each walk with its start.
  Near 1, the walk didn't move the chains; below about 0.3, it did. It often rises as $\beta$
  approaches 1, where the posterior is narrowest.

`parents`
: the distinct samples the resampling kept.

`lineages`
: the initial samples the population descends from. Few lineages are normal after many steps, and
  harmless when the walks decorrelate: runs that ended with 1-17 lineages were right.

`states`
: with a {ref}`pool <pooling>`, how many independent states each chain's kept states are worth.

The correlation is measured against the population's own spread, so it can't see a population that
collapsed as a whole: the 128-chain Metropolis run above had low correlations and a posterior less
than half as wide as it should be.

The direct test is to **repeat the run**. Two or three runs with the same settings, compared
parameter by parameter in units of the posterior standard deviation, measure what one run can't:

- the means of runs of independent samples scatter by about $1/\sqrt{N_\mathrm{eff}}$, i.e. 0.016
  for 4096 independent samples, 0.09 for 128;
- differences well above that come from walks that don't mix;
- for a quantity derived from the samples, e.g. the total potency or the moment magnitude, the
  effective samples of a run are $\sigma^2 / \mathrm{var}(\text{run means})$.

Given a reference, e.g. a longer run or the exact posterior of a linear problem, `diagnose` also
reports how far the run's means and spreads are from it.


## Choosing the settings

### The sampler

Metropolis needs no gradient and works well for tens of parameters. For hundreds, HMC is much
cheaper for the same accuracy: on the static case, decorrelating Metropolis walks took 320000 steps
(261 s) where HMC took 3400 trajectories (49 s), for the same posterior. Use HMC, or MALA, when the
model provides gradients.

### Step sizes

By default, the Metropolis scaling starts at $2.38/\sqrt{d}$ and the HMC step at $d^{-1/4}$ (MALA:
$d^{-1/3}$), $d$ the number of parameters, and both adapt to their target acceptance. Metropolis
adapts once per $\beta$ step, by at most a factor of 2, so its start matters; HMC adapts after every
trajectory. Set them only to reproduce an earlier run.

### The mass matrix (HMC)

Keep the default diagonal mass. Use a dense one only with at least twice as many chains as
parameters, and with walks that mix.

### The walks

Let a {ref}`decorrelating step counter <stepcounters>` choose the walk length: it walks until the
correlation with the start falls below a target, so that it spends its steps where they are needed,
a few at high temperature and many near $\beta = 1$. Its defaults are sized for Metropolis; for HMC,
give it its own:

| sampler | `min_mc_steps`, `corr_check_steps` | `max_mc_steps` | `target_correlation` | `statistic` |
|---|---|---|---|---|
| Metropolis | 200 | 10000 | 0.3 | `max` |
| HMC, tens of parameters | 5 | 200 | 0.3 | `max` |
| HMC, hundreds of parameters | 2-5 | 40-200 | 0.3 | `median` |

With hundreds of parameters, the largest correlation rarely falls below the target: a few slow
parameters keep the walks to their maximum. The median lets them stop when most have moved.

For HMC on a hard posterior, also set the **trajectory length** instead of the number of leapfrog
steps (`trajectory_length = 1.0`): the leapfrog steps then follow the step size, and a chain moves
about one posterior standard deviation per trajectory. Few long trajectories decorrelate the chains
for fewer gradients than many short ones: on the kinematic case, 25 trajectories of 40 leapfrog
steps mixed far better than 100 of 10. Pair it with a decorrelating counter: with a fixed number of
trajectories, the leapfrog steps grew to 133 per trajectory near $\beta = 1$, and that run cost three
times the cheapest.

### The chains

There is no agreed minimum. The number of chains a problem needs is the largest of four:

1. **For the sampler to adapt.** The Metropolis proposal is the weighted covariance of the
   population, whose effective size is about $N/2$ at a COV target of 1. On the static case, with
   walks long enough at every count, its posterior spreads came out at 0.67 of the right ones with
   $0.75d$ chains, $d$ the parameters, 0.87 with $1.5d$, 0.96 with $3d$, and 0.98-0.99 with $6d$-$12d$:
   use at least $3d$, and $6d$ or more for spreads within a few percent. An HMC diagonal mass has no
   such limit: from 64 chains ($0.2d$) to 4096, the runs scattered as independent samples of their
   size, and their spreads were within 1% from 128 chains on. A dense mass needs $N \ge 2d$, which
   the run enforces, and walks that mix.
2. **For the precision.** With walks that mix, a posterior mean is within about
   $1/\sqrt{N_\mathrm{eff}}$ of its posterior standard deviation: 0.09 for 128 chains, 0.03 for
   1024. Tails, e.g. a 95% interval, need about a thousand effective samples or more. A pool and a
   production walk add effective samples without more chains.
3. **For the modes.** To carry a mode of weight $p$ through the resamplings, $Np$ must stay well
   above 1: about $N \gtrsim 20/p$-$50/p$, e.g. 500-1000 chains for a mode of 5%.
4. **For the evidence,** much noisier than the posterior: thousands.

What worked on the cases here:

| problem | chains | limited by |
|---|---|---|
| tens of parameters, Metropolis (linear, 18) | 256 | precision |
| hundreds, Metropolis (static, 342) | $3d \approx 1000$ for spreads within 5%, $6d$-$12d$ within 2-3% | adaptation |
| hundreds, HMC with a diagonal mass (static, 342) | any, from 64 on: the error followed $1/\sqrt{N}$ | precision |
| hundreds, HMC with long walks (kinematic, 690) | 256 ($0.37d$); more didn't lower the scatter between runs, about 0.1 sd | the walks |
| a dense mass | $\ge 2d$, with long walks | adaptation |
| few parameters, an expensive model | a few hundred, with a pool and a production walk (untested) | precision |
| comparing evidences | thousands | the evidence's spread |

Beyond that, spend the budget on the walks. On the 690-parameter kinematic case, with long walks,
256, 512, 1024 and 2048 chains gave the same posterior, and two runs differed by about 0.1 posterior
standard deviations at every count: the rest of the error came from the walks, not the chains. 256
chains took 1 hour on one GPU, 2048 took 3.2; 4096 chains with short walks, in 2 hours, scattered
three to four times as much from run to run.

### The COV target

The default, 1, keeps half of the samples effective at each $\beta$ step. A larger target takes
fewer, larger steps. On the static case the walks took about as long at every step, so the cost
followed the number of steps:

| COV target | β steps | trajectories | posterior |
|---|---|---|---|
| 0.5 | 156 | 8985 | the same |
| 1 | 83 | 3995 | the same |
| 2 | 50 | 2733 | the same |

With thousands of chains and walks that mix, 1.5-2 saves about a third of the time. With few
chains, or if you need the evidence, stay at 1 or below.

### The resampling

The default, systematic resampling, gives each sample the floor or the ceiling of its expected
number of copies. Multinomial resampling, independent draws, loses more samples by chance: 128
distinct parents out of 256 at each step, against 161 for systematic, on the linear case.

### Pooling and the production walk

A {ref}`pool <pooling>` keeps several states of each chain's walk in the population. They count as
more samples only if they are far enough apart: about 1.1-1.3 states per chain from short walks,
2.4-3.4 from long decorrelating ones. On the static case, 128 chains with long walks and a pool of
16 beat the precision of 128 independent samples.

For the samples of the analysis, add a **production walk**: `production` more steps at
$\beta = 1$, without resampling and with the sampler fixed, the pool spread evenly over them. 1024
chains keeping 16 states a decorrelation time apart give about 16000 nearly independent samples.

A production walk can't move weight between modes; only the annealing does. For that, and for an
honest error bar, repeat the run: the results of independent runs can be pooled.


## Starting points

| problem | controller | settings |
|---|---|---|
| tens of parameters, cheap forward model | `altar.bayesian.catmip` | thousands of chains, a few hundred steps, or decorrelating walks |
| a static slip inversion, hundreds of parameters | `altar.bayesian.catmip_hmc` | 4096 chains, 20-50 trajectories, or decorrelating walks; COV 1-2 |
| a kinematic inversion, hundreds of correlated parameters | `altar.bayesian.catmip_hmc` | 256-512 chains; `trajectory_length = 1`; decorrelating walks on the median; a pool of 16 spread over the walks; a production walk |

For the kinematic case, in full:

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
        production = 160
    job.chains = 2**9
```

For a model with few parameters and an expensive forward problem, e.g. an earthquake cycle model,
the chains, not the walks, are the cost: use as few as the precision needs, and a pool with a
production walk for the samples. This has not been tested as thoroughly as the cases above.


## The evidence

The log evidence printed at the end of a run matters only to compare models of the same data, or an
algorithm against an exact answer. It assumes the chains start from the prior, a fixed likelihood,
and normalized priors; see {ref}`the COV scheduler <cov-scheduler>`.


## A checklist

1. Use HMC for hundreds of parameters, with the default diagonal mass.
2. Let a decorrelating step counter choose the walks, with numbers for the sampler you use.
3. Run `diagnose`: are the walk correlations low at every $\beta$ step, including the last ones?
4. Repeat the run once or twice: do the means agree within a few hundredths of a standard deviation
   for thousands of chains, a tenth for hundreds?
5. Check the quantities you report, e.g. the moment magnitude, across the runs.
6. For the final samples, add a production walk.
