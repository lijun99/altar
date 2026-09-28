(quickstart)=
# Quick Start

We use the linear model to show how to run a Bayesian inversion with AlTar. The steps are:

1. prepare a configuration file, e.g. `linear_catmip.pfg`, with the settings of the simulation;
2. prepare the input data files the model needs: for the linear model, the observed data, their
   covariance, and the Green's functions;
3. run the AlTar application of the model, `altar-linear`;
4. collect and analyze the results.

The example comes with AlTar, in `models/linear/examples`; the
{doc}`linear model tutorial <../tutorials/linear/linear>` walks through it as a notebook.

## Prepare the configuration file

A configuration file passes the settings to an AlTar application. For the linear model:

```{literalinclude} ../../models/linear/examples/linear_catmip.pfg
:language: none
:caption: linear_catmip.pfg
:lines: 10-
```

`.pfg` (pyre configuration) files are human-readable and structured by indentation, much like YAML;
a setting can also be given by its full or partial path, e.g. `job.tasks` (see {doc}`Pyre`).

The root, `linear`, is the name of the application. Its components are

- `model`: the model's own settings, such as the prior distributions of its parameters, the
  parameters of the forward model, and the observed data;
- `controller`: how the posterior is sampled; here `altar.bayesian.catmip`, CATMIP annealing
  with Metropolis sampling;
- `job`: the size of the simulation and how it runs, e.g. the number of chains and of steps per
  $\beta$, cpu or GPU, one or several processes.

```{note}
Any component or setting left out of the configuration keeps its default.
```

The model settings depend on the model: see the model's page in {doc}`Models`. The controller and
the job are described in {doc}`AlTarFramework`.

## Prepare the input files

Small settings go into the configuration file; large data sets come from files. For the linear
model, the data likelihood is

$$
P(\mathbf d|\boldsymbol\theta) = \frac{1}{\sqrt{(2\pi)^n \det C_d}}
\exp\left[-\frac{1}{2} \left(\mathbf d - \mathbf d^{pred}\right)^T C_d^{-1}
\left(\mathbf d - \mathbf d^{pred}\right)\right],
$$

where $\boldsymbol\theta$ holds the $m$ parameters, $\mathbf d$ the $n$ observations, and the
$n \times n$ covariance matrix $C_d$ the uncertainties of the data. The prediction comes from the
forward model,

$$
\mathbf d^{pred} = \mathbf G \boldsymbol\theta,
$$

with the $n \times m$ matrix of Green's functions $\mathbf G$.

So the model needs three inputs, $\mathbf d$, $C_d$ and $\mathbf G$, here as the text files
`data.txt`, `cd.txt` and `green.txt` (`.h5` and raw binary `.bin` files work too). They are read
from the directory `model.case`, `patch-9` in the example; the file names are set by
`model.dataobs.data_file`, `model.dataobs.cd_file` and `model.green`. Instead of a covariance
file, a constant standard deviation for all observations can be given with
`model.dataobs.cd_std`.

## Run the application

Each model comes with its application; for the linear model, `altar-linear`, a short python
script:

```{literalinclude} ../../models/linear/bin/altar-linear
:language: python
:lines: 11-
```

It defines the `Linear` application, whose default model is the linear model, and runs it. Run it
in the directory with the configuration file and the `case` directory:

```bash
cd models/linear/examples
altar-linear --config=linear_catmip.pfg
```

Any setting can be changed on the command line, e.g. the number of chains, or the backend:

```bash
altar-linear --config=linear_catmip.pfg --job.chains=2**10
altar-linear --config=linear_catmip.pfg --job.gpus=1
```

See {doc}`AlTarFramework` for the other run options.

## Collect and analyze the results

The run prints the posterior mean and standard deviation of each parameter at every $\beta$ step,
and writes the history of the annealing to `results/BetaStatistics.txt`. To keep the samples of
each $\beta$ step, save them to HDF5 files with the `h5recorder` archiver:

```bash
altar-linear --config=linear_catmip.pfg --controller.archiver=altar.bayesian.h5recorder
```

which writes `results/step_nnn.h5` for the $\beta$ steps, and `results/step_final.h5` for the
posterior. Each file holds

```none
step_nnn.h5
├── Annealer
│   ├── beta                       ; the β of the step
│   └── weights                    ; the importance weights of the samples
├── Bayesian
│   ├── prior                      ; log prior of each sample, (samples)
│   ├── likelihood                 ; log data likelihood of each sample
│   └── posterior                  ; log posterior of each sample
├── ParameterSets
│   ├── <pset>_sampling            ; the samples of each parameter set, (samples, parameters)
│   ├── <pset>_physical            ; ... in physical space, for reparameterized parameter sets
│   ├── jacobian                   ; log|J| of each sample, for reparameterized parameter sets
│   └── has_reparametrization
├── Proposal
│   └── sigma                      ; the covariance of the Metropolis proposal
└── Statistics
    └── mc_updates                 ; the acceptance statistics of each β step
```

where `<pset>` is the name of each parameter set, `all` in the example. The files can be viewed
with HDFView, or read with `h5py`, e.g. for the posterior mean and standard deviation:

```python
import h5py

with h5py.File("results/step_final.h5", "r") as h5:
    theta = h5["ParameterSets/all_sampling"][:]
print(theta.mean(axis=0), theta.std(axis=0))
```

The probabilities are saved as logarithms. For a well-constrained problem with a Gaussian
posterior, like this one, the posterior mean is the estimated solution, and the standard deviations
its uncertainties.
