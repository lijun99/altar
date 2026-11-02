(optimization)=
# The best fit

Sampling the posterior answers what the data say about the parameters. Two simpler questions often
come first, or alongside: how well can this model fit the data at all, and where? The `optimize`
action answers them at a fraction of the cost of a sampling run: it finds the maximum of the data
log likelihood, or of the posterior, without an annealing schedule or the mixing of walks to worry
about. This chapter shows how to use it; its settings are listed in {ref}`the best fit <optimize>`.

Use it to:

- check what a model can do before sampling it: whether it fits the data at all, and whether a
  change to the model, more free parameters or another fixed input, improves the best fit, and by
  how much;
- check a posterior: a posterior whose likelihoods sit well below the best fit has not found the
  region of the best fit, the sign of walks that did not mix;
- get a starting point, or a `theta` for the other actions, e.g. the {ref}`forward check
  <forward-check>`;
- get a quick estimate of the uncertainties, from the curvature at the optimum, where the posterior
  is close to gaussian.

## A first run

The actions run through a *plexus*, an application that takes the action as its first argument.
Any model has one with a three-line script, named after the application so that its settings apply;
for the mogi example, in its `examples` directory, `mogi.plexus`:

```python
#!/usr/bin/env python3
import altar, altar.models.mogi
raise SystemExit(altar.shells.altar(name="mogi").run())
```

The example's configuration, `mogi.pfg`, relies on the default model of `altar-mogi`; the plexus
needs it named, on the command line or as `model = altar.models.mogi` in the `mogi` section. With
the synthetic data made first, as the configuration explains,

```bash
python3 mogi.plexus optimize --config=mogi.pfg --model=altar.models.mogi
```

runs CMA-ES, the default method, from the means of the priors, and reports:

```none
 --   start data log likelihood -12275.962
 --   stopped after 193 generations, 1738 evaluations: the best value changed less than 1e-06 in 30 generations
 --   best data log likelihood 529.90484, +12805.9 from the start
 --   parameter set    count          mean                    [min, max]  on a bound
 --   location             2  -0.000268253   [-0.000748367, 0.000211861]           0
 --   depth                1          3000                  [3000, 3000]           0
 --   source               1         1e+07                [1e+07, 1e+07]           0
 --   offsets              2   4.26547e-08    [3.45457e-08, 5.07636e-08]           0
```

the source of the synthetic data, at (0, 0), 3000 m deep, with a volume change of $10^7$ m³. The
best point is in `optimize.h5`, with the history of the search, and in `optimize.txt`, one row of
parameters.

## Polishing, and the uncertainties at the optimum

CMA-ES is robust far from the optimum, and slow to settle its last digits. Levenberg-Marquardt,
`method = lm`, is the opposite, and gives the covariance at the optimum, $(J^T J)^{-1}$, from the
jacobian of the whitened residuals. The two together, CMA-ES to find the optimum and
Levenberg-Marquardt from its best point:

```bash
python3 mogi.plexus optimize --config=mogi.pfg --model=altar.models.mogi \
    --mogi.optimize.method=lm --mogi.optimize.theta=optimize.txt --mogi.optimize.output=polish.h5
```

```none
 --   stopped after 5 iterations, 94 evaluations: no step improved the fit
 --   parameter set    count          mean                    [min, max]  on a bound   median sd
 --   location             2  -6.38987e-14    [-2.2128e-13, 9.34821e-14]           0       25.41
 --   depth                1          3000                  [3000, 3000]           0       49.24
 --   source               1         1e+07                [1e+07, 1e+07]           0   3.113e+05
 --   offsets              2  -5.05275e-20   [-1.98443e-18, 1.88337e-18]           0     0.00138
```

`polish.h5` holds the standard deviation of each parameter, in `best/sd`. Where the posterior is
close to gaussian, they are its standard deviations; here, against a CATMIP run of the same model,
`altar-mogi --config=mogi.pfg`, 4096 chains:

| parameter | optimum | sd at the optimum | posterior mean | posterior sd |
|---|---|---|---|---|
| x [m] | 0 | 28.3 | 0.52 | 28.1 |
| y [m] | 0 | 22.5 | 0.21 | 22.9 |
| depth [m] | 3000 | 49.2 | 3004 | 49.0 |
| volume change [m³] | 1.000e7 | 3.11e5 | 1.003e7 | 3.10e5 |
| offset east [m] | 0 | 0.00097 | 0.00004 | 0.00097 |
| offset west [m] | 0 | 0.00179 | 0.00008 | 0.00178 |

The posterior means are within a tenth of a standard deviation of the optimum, and the standard
deviations agree to 2%. When they don't, the posterior is not gaussian: the model is nonlinear over
the posterior, a bound cuts into it, or it has more than one mode, and the sampled posterior is the
answer.

With `posterior = yes`, the action maximizes the posterior under the conjugate prior, the gaussian
matched to each parameter set's prior, instead of the data log likelihood: the maximum a posteriori
point, and, with `lm`, the Laplace approximation of the posterior. Without it, the prior only
bounds the search, through the supports of the priors.

## Choosing a method

| method | needs | use for |
|---|---|---|
| `cma` | the forward model only | the start, far from the optimum, a model with several optima, up to a few hundred parameters |
| `lm` | the forward model; a jacobian by finite differences, $2n$ runs per iteration | the polish near an optimum, and the uncertainties there |
| `lbfgs` | the same jacobian | an alternative to `lm`; slower on these least-squares problems, and in its last digits |

All three keep the parameters inside the supports of their priors, and report how many of them end
on a bound. `lm` and `lbfgs` measure their steps in units of `step` times the prior standard
deviations, so that parameters of very different scales, e.g. a depth in meters and a volume change
in cubic meters, are treated alike. For a linear model with a gaussian prior, the posterior is
known exactly, and the action is a check rather than a tool.

How many generations CMA-ES needs grows with the number of parameters, and with how badly they are
scaled and correlated: about 200 for the 6 parameters of mogi, about 500 for an 18-parameter linear
problem. A larger `population` searches more globally, and costs nothing more per generation while
it fits in the batch the model runs at once.

## Narrower bounds, and fixed parameters

The bounds of the search are the supports of the priors, so a narrower range for a run is a
narrower `prior.support` in its configuration, e.g. in a copy kept for the optimization. A support
belongs to a whole parameter set: to bound one parameter differently, give it a set of its own.

To hold parameters fixed, list them in `fixed`: a whole set, an entry or a slice of one, at its
start value or at a value given with it. For mogi, the polish of the best fit with the offsets at
0, the source at 3500 m instead of the 3000 m of the truth, and its second coordinate where the
start has it, in the `mogi` section:

```yaml
mogi:
  optimize:
    method: lm
    theta: optimize.txt
    output: fixed.h5
    fixed:
      - offsets=0          # the whole set, at 0
      - depth=3500         # at 3500 m
      - location[1]        # the second entry of the set, at its start value
```

or on the command line, `--mogi.optimize.fixed="[offsets=0, depth=3500, location[1]]"`. Indices
count from 0 within the set, `name[i:j]` is a slice that excludes `j`, and a value has to be inside
the support of the prior. In a yaml flow list, `[...]`, quote the entries with brackets. The run
reports:

```none
 --   fixed: 4 parameters (offsets=0, depth=3500, location[1]), 2 free
 --   start data log likelihood 82.793719
 --   stopped after 8 iterations, 86 evaluations: no step improved the fit
 --   best data log likelihood 410.59621, +327.802 from the start
 --   parameter set    count          mean                    [min, max]  on a bound   fixed   median sd
 --   location             2      -14.6067      [-29.2126, -0.000748367]           0       1       22.11
 --   depth                1          3500                  [3500, 3500]           0       1           -
 --   source               1     1.192e+07        [1.192e+07, 1.192e+07]           0       0   7.505e+04
 --   offsets              2             0                        [0, 0]           0       2           -
```

The deeper source needs a larger volume change, $1.19 \times 10^7$ m³ instead of $10^7$, and still
fits 119 worse in log likelihood than the free best fit, 529.9: the data rule out 3500 m. The
optimizer only sees the free parameters, so a fixed parameter costs `lm` and `lbfgs` no finite
differences. The fixed parameters keep their values in `best/theta`, with zero standard deviations
in `best/sd`, and the output marks them in `fixed`.

With fixed parameters, `lm` gives the conditional optimum and the conditional uncertainties of the
others: how well they are determined if the fixed ones are known. Where they are correlated with
the fixed ones, these are smaller than with everything free: the standard deviation of the volume
change is $7.5 \times 10^4$ m³ with the depth fixed, against $3.1 \times 10^5$ with it free. On the
linear example, a fixed run matches the conditional mean and standard deviations of the exact
posterior to $10^{-6}$. To compare two variants by their best fit, e.g. a parameter fixed against
free, the difference of their best log likelihoods says how much the fit gains from freeing it.

## Expensive models: MPI and GPUs

Each step runs the forward model on a batch of points: a generation of CMA-ES, the $2n$ points of a
jacobian, or the trial steps of `lm` and `lbfgs`. Under MPI, the batch is shared among the
processes, as in a sampling run:

```bash
python3 mogi.plexus optimize --config=mogi.pfg --model=altar.models.mogi \
    --shell=mpi.shells.mpirun --job.tasks=4
```

and, for a model that runs on gpus, with `--job.gpus=1`, each process on its own gpu (see
{ref}`the job <job>`). The first process runs the optimizer, sends each batch, and collects the
results; the result is the same as in one process. The plexus must be a script, as above, since
`mpirun` starts it again in each process; and pyre must find the MPI that runs it (see
{ref}`MPI <installation-mpi>`). To keep every gpu busy, make the population a multiple of the
processes, e.g. 32 points per gpu, and set `batch` if a process can't run its whole share at once.

## When the best fit is not good enough

The best fit is the most a model can do. A best fit far from the data, by the χ² of the {ref}`forward
check <forward-check>`, is a model that is missing something, and sampling it more carefully won't
help. Two signs to look for:

- **Parameters on their bounds.** A best fit that pushes many parameters onto the bounds of their
  priors is compensating: the data ask for something the model can only approximate by its extreme
  values. Widening the bounds may improve the fit, but look first at what the model lacks.
- **The fit improves with every new freedom, but stays far from the data.** Freeing more
  parameters, one set after another, and comparing the best fits tells how much each freedom is
  worth. If none of them closes most of the gap, the misfit is in what all the variants share: a
  fixed input, the physics, or the data and their processing.

Each comparison costs one optimization, not one sampling run; sample the variants that survive.

## Common problems

- *The plexus uses the null model.* Name the model, with `--model=altar.models.<name>` or `model =`
  in the application's section.
- *Under MPI, every process fails at once.* `mpirun` starts the plexus again from its command line;
  `python -c "..."` doesn't survive that: use a script.
- *CMA-ES stops early, "the covariance became ill-conditioned".* The parameters are fit along a
  direction the data don't constrain; fix those parameters, or give them a prior with `posterior`.
- *`lm` stops at once, "no step improved the fit".* The start is already at the optimum, to the
  precision of the finite differences, or far from it in a region the linearization doesn't
  describe: start from the best point of `cma`.
- *A $C_p$ that depends on the parameters* is computed once, at the start, and held fixed.
