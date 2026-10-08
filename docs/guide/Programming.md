(programming-guide)=
# Programming Guide

This guide shows how to write a model for your own inverse problem.

AlTar's framework is written in python; the compute-intensive parts are C++ or CUDA, bound to
python as extension modules with [pybind11](https://pybind11.readthedocs.io). It follows the
component model of pyre (see {doc}`Pyre`): a model is a component, whose settings and parts are
configured at run time, and pyre deploys the simulation to where it runs, one process or several,
cpu or GPU.

## A first model

The simplest model is all python, on the cpu: the linear regression model in `models/regression`,
which fits data $(x_n, y_n)$ with a line,

$$
y = \text{slope} \times x + \text{intercept} + \epsilon,
$$

sampling the slope and the intercept.

```{literalinclude} ../../models/regression/regression/Linear.py
:language: python
:pyobject: Linear
```

It builds on the `BayesianL2` template, which provides everything but the forward model:

- the parameter sets (`psets`): the initial samples, the priors, and the check of proposals
  against bounded priors;
- the observed data (`dataobs`): loading the data and their covariance, and the data likelihood
  with the L2 norm;
- the posterior, and the reparameterization of bounded priors.

The model only adds the input file with the $x_n$, which `initialize` loads with the model's file
reader, `self.io.load`, finds its parameters in a sample from the offsets of the parameter sets,
and computes the predictions of one sample in `forward_model`. It also provides `forward_problem`,
for the {ref}`forward check <forward-check>`, and `gradient`, for the gradient-based samplers.
Its configuration, `models/regression/examples/linear.pfg`:

```{literalinclude} ../../models/regression/examples/linear.pfg
:language: none
:start-at: "regression:"
```

The $y_n$ are the observed data, `dataobs`; the parameter sets, `slope` and `intercept`, give the
layout of $\boldsymbol\theta$, one row per sample. Run it with

```bash
cd ~/tools/src/altar/models/regression/examples
altar-regression --config=linear.pfg        # CATMIP
altar-regression --config=linear_hmc.pfg    # CATMIP with HMC, through the gradient
```

## The model protocol

The framework calls a model through the `Model` protocol:

```{literalinclude} ../../altar/altar/models/Model.py
:language: python
:pyobject: Model
```

`initialize` receives the application, the root component, to read the job (the number of chains,
the backend) and the other components. Most of the other methods receive a `step`, which holds the
state of the chains:

- `theta`, the samples, one row per chain;
- `prior`, `data`, `posterior`: the log densities of each sample;
- `beta`, the current $\beta$.

The densities are logarithms, and AlTar processes the chains as a batch: a model evaluates all the
samples of a step at once.

## The `BayesianL2` template

With `BayesianL2`, a model writes its forward model, and whatever else it needs of the following.

### The forward model

`forward_model(theta, prediction)`
: the prediction, or the residual, for a single sample; on the cpu, both are numpy vectors.

`forward_model_batched(theta, prediction, batch=None)`
: the predictions for the first `batch` samples at once, a (samples × observations) matrix. The
  default calls `forward_model` for each sample; override it when a batch is faster, e.g. a matrix
  product for a linear model, or a GPU kernel.

`return_residual`
: whether the forward model returns the residuals, prediction − observation (`True`, the default),
  or the predictions, from which the residuals are then computed.

### The observed data

`dataobs` loads the observed data and their covariance, and computes the data likelihood,

$$
\log P(\mathbf d | \boldsymbol\theta) = -\frac{1}{2}
\left(\mathbf d^{pred} - \mathbf d\right)^T C_\chi^{-1} \left(\mathbf d^{pred} - \mathbf d\right)
+ c,
$$

where $c$ normalizes it, from the determinant of $C_\chi$. It holds the Cholesky factor of
$C_\chi^{-1}$, as `dataobs.cd_inv`, a scalar for a constant variance. For a linear model, merging
that factor into the Green's functions once, $\mathbf G \leftarrow L^T \mathbf G$, saves a
triangular product at every evaluation; the GPU implementation of the linear model does so. Its
settings are listed with the {doc}`static model <Static>`.

### The check against the data

`forward_problem(application, theta)`
: the raw predictions for each row of `theta`, a (samples × parameters) numpy array in physical
  space, as a dictionary of numpy arrays: `"data"`, (samples × observations), and anything else
  worth saving, e.g. the slip history of the kinematic model. The
  {ref}`forward check <forward-check>` calls it.

### Gradients

The gradient-based samplers, HMC, MALA and SGLD, need

`gradient(controller, step, batch=None)`
: fills the gradients of the log prior and of the log data likelihood with respect to `step.theta`:
  `step.grad_prior` and `step.grad_data` on the cpu, `step.prior_gradient` and
  `step.data_gradient` on the GPU, with respect to the physical parameters. The prior gradient
  comes from the parameter sets (`prior_gradient`); for reparameterized priors, the framework
  takes both into sampling space.

The regression model computes it with numpy:

```{literalinclude} ../../models/regression/regression/Linear.py
:language: python
:pyobject: Linear.gradient
```

and the linear model with numpy on the cpu, and with cuBLAS on the GPU.

### Model uncertainty

A model that can estimate its uncertainty $C_p$ (see {doc}`StaticCp`) provides

`compute_cp(theta)`
: $C_p$, (observations × observations), for the mean model `theta`;

`covariance_updated()`
: redo whatever depends on the data covariance, once it changed, e.g. the Green's functions merged
  with it.

The model's `cp` component decides when to call them; `update_covariance(cp)` sets the new
covariance.

## cpu and GPU implementations

A model that runs on both the cpu and the GPU keeps its numerics apart from its component: the
component declares the settings, and, at `initialize`, builds an implementation for the active
backend, a plain python class in its `native` or `cuda` subpackage, then forwards the work to it.
The linear model:

```{literalinclude} ../../models/linear/linear/Linear.py
:language: python
:pyobject: Linear._makeImpl
```

```{literalinclude} ../../models/linear/linear/Linear.py
:language: python
:pyobject: Linear.forward_model_batched
```

`altar.backends.active()` is `"cuda"` when the job runs on a GPU (`job.gpus = 1`). The
distributions follow the same pattern; one defined in a model's package points `impl_package` at
it, e.g. `impl_package = "altar.models.seismic"` for the moment magnitude prior.

### Arrays on the GPU

On the GPU, the steps hold their arrays in CUDA managed memory, `altar.cuda.matrix` and
`altar.cuda.vector`:

```python
import numpy
import altar.cuda

θ = altar.cuda.matrix(shape=(samples, parameters), dtype="float32")
numpy.asarray(θ)[:, :] = 0      # a numpy view of the same memory
```

A numpy view reads and writes the same memory, which is convenient for setting up, but runs on the
cpu: keep it out of the hot paths, and compute with cuBLAS (`altar.cuda.cublas`, e.g. `dgemm`,
`sgemm`, `dtrmm`, `axpy`, with the handle `altar.cuda.cublas_handle()`) or your own kernels. A
binding takes the underlying pyre grid, `θ.grid`.

### CUDA kernels

A model's kernels live in its `lib/libcuda<model>` (a shared library) and its bindings in
`ext/cuda<model>` (a pybind11 module). A kernel receives the samples as pyre grid views, one thread
per sample, e.g. the log density of the moment magnitude prior:

```{literalinclude} ../../models/seismic/lib/libcudaseismic/cudaMoment.h
:language: c++
:start-at: "namespace altar::models::seismic::cudaMoment"
:end-before: "// end of file"
```

and the binding hands the python grids to it, dispatching on their precision:

```{literalinclude} ../../models/seismic/ext/cudaseismic/moment.cc
:language: c++
:start-at: "    // likelihood[s] += the moment"
:end-before: "    // gradient[:, idx_begin:idx_end] <- d/dtheta"
```

`regrid` views a python grid as a pyre grid of the given cell type and rank. The kernel runs
asynchronously: pyre's managed grids wait for the device whenever python reads their cells, so a
binding returns as soon as it has launched. `synchronize` only checks the launch, or waits for the
kernel when the environment variable `ALTAR_CUDA_SYNC` is set, which pins an error on the kernel
that caused it. See `models/seismic` for the whole of it: the library, the module, and their
build.

### cuTile kernels

A kernel can also be written in python, with NVIDIA's [cuTile](https://docs.nvidia.com/cuda/cutile-python)
(`cuda.tile`): a kernel works on tiles, blocks of an array of a fixed shape, and cuTile compiles it
the first time it runs. The volcano models (see {doc}`Volcano`) have their GPU forward models
written this way, with no C++ and no extension module for the GPU. The Mogi kernel computes a tile
of samples × observations:

```{literalinclude} ../../models/mogi/mogi/CUDA.py
:language: python
:pyobject: displacements
```

and `altar.cuda.tile.launch` runs it, with the arrays of the steps:

```{literalinclude} ../../models/mogi/mogi/CUDA.py
:language: python
:pyobject: CUDA.forward_model_batched
```

`launch` hands the arrays to cuTile, and runs the kernel on the default stream, in order with the
rest of AlTar's kernels. Two things to keep in mind:

- python floats, as arguments of a kernel or literals in it, are single precision in cuTile:
  keep the constants that need double precision in a device vector, and read them with
  `altar.cuda.tile.constant(constants, index)`;
- cuTile inlines the functions a kernel calls, and unrolls the loops over python tuples: a large
  kernel can take very long to compile. A loop over `range(n)` stays a loop; the CDM kernel keeps
  the twelve sides of its dislocations in a table, and loops over them.

## Ensembles of models

An ensemble, `altar.models.ensemble`, combines models that share parameters: it owns the parameter
sets, and each model computes its data likelihood from its own columns of $\boldsymbol\theta$ (see
{doc}`Kinematic`). A model that works in an ensemble only needs its `parameters`, which the ensemble
sets, and its own `psets_list`, with the names of the ensemble's parameter sets it uses.

## Checking a sampler

The linear example has an exact posterior: its data are linear in the parameters, with Gaussian
noise, so under a Gaussian prior the posterior is Gaussian, known in closed form.
`models/linear/tests/posterior.py` runs the samplers on it and compares the mean, the standard
deviations and the correlations of their final samples with the exact ones:

```bash
cd ~/tools/src/altar/models/linear/tests
python posterior.py --list          # the cases: controllers, samplers, priors
python posterior.py                 # all of them, on the cpu
python posterior.py mala --gpu      # one, on the GPU
```

The tolerances allow for the noise of 256 chains; they catch errors of the size of a wrongly
handled prior, not biases much below 0.3 posterior standard deviations. A new sampler should
pass it, on both backends, with and without a reparameterized prior.

## Code organization

A model is a python package under `models/`, built and installed with the rest of AlTar. For a
model named `regression`:

```none
models/regression
├── CMakeLists.txt          # builds and installs the package and its application
├── bin
│   └── altar-regression    # the application
├── regression              # the python package, installed as altar.models.regression
│   ├── __init__.py         # its foundries, e.g. altar.models.regression.linear
│   ├── meta.py.in          # its version, filled in by the build
│   └── Linear.py           # the model
└── examples
    ├── linear.pfg          # an example configuration
    ├── linear_hmc.pfg      # the same, with HMC
    └── synthetic           # its input files
```

A model with cpu and GPU implementations adds, like `models/linear` and `models/seismic`,

```none
models/seismic
├── seismic
│   ├── Static.py           # the components
│   ├── native/             # the cpu implementations
│   ├── cuda/               # the GPU implementations
│   └── ext/                # hosts the extension module
├── lib/libcudaseismic      # the CUDA kernels, a shared library
└── ext/cudaseismic         # their pybind11 bindings, the module cudaseismic
```

For CMake, the build functions of a model go in `.cmake/altar_<model>.cmake` at the root of the
source tree (e.g. `.cmake/altar_seismic.cmake`), and the model into the root `CMakeLists.txt`, with
`add_subdirectory(models/<model>)`. For mm, a model is a project: add it to `.mm/projects.mm`, and
describe its package, library and extension module, with their source directories, in
`.mm/<model>.mm` (e.g. `.mm/mogi.mm`; `.mm/seismic-cuda.mm` for the GPU parts of a model, built
only when mm finds CUDA). mm builds the sources of an extension module, but its entry point
`<module>.cc`, into a library: keep the bindings in a file of their own, e.g.
`models/mogi/ext/mogi/bindings.cc`.

## Data types

### Configurable properties

The traits of a component are declared with `altar.properties`, e.g. `int`, `float`, `bool`,
`str`, `path`, `array`, `list`, `dict`, and with the protocols of other components, e.g.
`altar.distributions.distribution()`. Each takes a `default`, and a `doc`.

### Arrays on the cpu

On the cpu, the samples and the densities are numpy arrays: `step.theta` is
(samples × parameters), and `step.prior`, `step.data` and `step.posterior` are (samples,). Other
parts of the framework hold on to them, so fill them in place rather than replace them:

```python
step.data[...] = 0                                   # zero them
step.posterior[:] = step.prior + step.beta * step.data
θ = step.theta[:, offset:offset + count]             # a view of some columns
```

The `altar.cuda` arrays take the same whole-array assignments, `a[...] = b` and `a[...] = 0`, so
code that runs on both backends can use them. The random numbers come from the numpy generator of
the application's `rng` component, `application.rng.rng`, seeded by its `seed`.
