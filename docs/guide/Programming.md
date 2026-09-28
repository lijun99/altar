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

The model only adds the input file with the $x_n$, loads it in `initialize` with the model's file
reader, `self.io.load`, and computes the residuals of one sample in `forward_model`. Its
configuration, `models/regression/examples/linear.pfg`:

```{literalinclude} ../../models/regression/examples/linear.pfg
:language: none
:start-at: "regression:"
```

The $y_n$ are the observed data, `dataobs`; the parameter sets, `slope` and `intercept`, give the
layout of $\boldsymbol\theta$, one row per sample.

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
: the prediction, or the residual, for a single sample.

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

The gradient-based samplers, HMC and SGLD, need

`gradient(controller, step, batch=None)`
: fills the gradients of the log prior and of the log data likelihood with respect to `step.theta`:
  `step.grad_prior` and `step.grad_data` on the cpu, `step.prior_gradient` and
  `step.data_gradient` on the GPU, with respect to the physical parameters. The prior gradient
  comes from the parameter sets (`prior_gradient`); for reparameterized priors, the framework
  takes both into sampling space.

See the `gradient` of the linear model for an example.

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

`regrid` views a python grid as a pyre grid of the given cell type and rank; `synchronize` waits
for the kernel, so that python reads its results. See `models/seismic` for the whole of it: the
library, the module, and their build.

## Ensembles of models

An ensemble, `altar.models.ensemble`, combines models that share parameters: it owns the parameter
sets, and each model computes its data likelihood from its own columns of $\boldsymbol\theta$ (see
{doc}`Kinematic`). A model that works in an ensemble only needs its `parameters`, which the ensemble
sets, and its own `psets_list`, with the names of the ensemble's parameter sets it uses.

## Code organization

A model is a python package under `models/`, built and installed with the rest of AlTar. For a
model named `regression`:

```none
models/regression
├── CMakeLists.txt          # builds and installs the package and its application
├── bin
│   └── regression          # the application
├── regression              # the python package, installed as altar.models.regression
│   ├── __init__.py         # its foundries, e.g. altar.models.regression.linear
│   ├── meta.py.in          # its version, filled in by the build
│   └── Linear.py           # the model
└── examples
    ├── linear.pfg          # an example configuration
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
only when mm finds CUDA).

## Data types

### Configurable properties

The traits of a component are declared with `altar.properties`, e.g. `int`, `float`, `bool`,
`str`, `path`, `array`, `list`, `dict`, and with the protocols of other components, e.g.
`altar.distributions.distribution()`. Each takes a `default`, and a `doc`.

### Matrices and vectors on the cpu

On the cpu, the samples and the densities are [GSL](https://www.gnu.org/software/gsl/) matrices
and vectors, `altar.matrix` and `altar.vector`, row-major:

```python
m = altar.matrix(shape=(rows, cols))   # rows x cols
m.zero()
m.fill(1.0)
c = m.clone()
c.copy(m)
```

They work with numpy through views, which share the memory:

```python
import numpy

view = numpy.asarray(m)      # changes to the view change m
copy = numpy.array(m)        # a copy
```
