(common-issues)=
# Common Issues

## Installation

### `import altar` fails after the installation

```none
ModuleNotFoundError: No module named 'altar'
```

pyre and AlTar install their python packages under `$CMAKE_INSTALL_PREFIX/packages`, which python
doesn't search by itself. With conda, make `$CONDA_PREFIX/packages` a link to the environment's
`site-packages` (see {doc}`Installation`); otherwise, add the `packages` directory to `PYTHONPATH`.

### AlTar builds without GPU support

```none
CUDA Toolkit or Pyre cuda extension not found; set WITH_CUDA to OFF
```

AlTar builds its GPU support only when it finds both a CUDA toolkit and a pyre built with CUDA.
Build pyre with `-DWITH_CUDA=ON`, which it needs explicitly, and make sure CMake finds `nvcc`: put
the toolkit's `bin` on the `PATH`, or set `CUDACXX`, e.g. `CUDACXX=/usr/local/cuda/bin/nvcc`.

### The build runs out of memory

Compiling the CUDA sources takes a lot of memory; with many parallel jobs, the machine can run out
of it, and under WSL2 the whole virtual machine may crash. Build with fewer jobs, e.g.
`cmake --build build -j 2`.

## Running

### Bad case name

```none
altar: bad case name: 'patch-9'
```

The application can't find the directory with the input files, `model.case`, usually because it
doesn't run where the configuration file and the input files are. Run it in that directory, and
name the configuration file with `--config`, e.g.

```bash
altar-linear --config=linear_catmip.pfg
```

### Configuration parser errors

```none
File ".../pyre/parsing/Scanner.py", line 71, in pyre_tokenize
    match = stream.match(scanner=self, tokenizer=tokenizer)
```

The configuration file is malformed; most often, it contains TAB characters, which `.pfg` files
don't allow. Replace them with spaces (see {ref}`the configuration files <pyre-config-format>`).

### YAML configuration files are ignored

```none
could not locate support for 'yaml'
```

pyre reads `.yaml` configuration files only with PyYAML installed; use `.pfg` files, or install
`pyyaml`.

### Gradient-based samplers and bounded priors

```none
gradient-based samplers (SGLD, HMC) only support unbounded priors; found bounded prior(s): Uniform.
Use CATMIP/Metropolis for this model, or set reparameterize=True on the prior.
```

HMC and SGLD need priors defined on the whole real line. Reparameterize the bounded priors it
names (`reparameterize = True`, see {doc}`Priors`), or sample with Metropolis.

### The kinematic model needs a GPU

```none
the kinematic model has only a cuda implementation; run it with job.gpus >= 1
```

Run it with `--job.gpus=1`, on a machine with an NVIDIA GPU and an AlTar built with GPU support.

### The covariance is not positive definite

```none
the data covariance C_chi is not positive definite: its Cholesky factorization failed at row ...
```

The data covariance, plus $C_p$ if the model has one, must be positive definite. Check the
covariance file; for a large or ill-conditioned covariance, factor it in double precision,
`dataobs.cd_dtype = float64`.

### The chains stop moving

The acceptance rate, in the log or in `BetaStatistics.txt`, drops to zero, and the run may stop with

```none
RuntimeError: spotrf: the leading minor of order 3 is not positive definite
```

when the samples collapse onto a few points and their covariance becomes singular. On the GPU in
single precision (`job.gpuprecision = float32`), try `float64` first: the cascaded static-kinematic
inversion, for one, needs it (see {doc}`Kinematic`). Otherwise, more steps per $\beta$
(`job.steps`) or more chains may help.

### MPI runs hang, or can't find `mpirun`

```none
AttributeError: 'NoneType' object has no attribute 'launcher'
```

pyre didn't find an MPI; or, with more than one MPI on the machine, the run hangs at the start,
because pyre launched the wrong `mpirun`. Tell pyre which MPI to use, and give its `mpirun` as a
full path, in `~/.config/pyre/mpi.pfg` (see {ref}`MPI <installation-mpi>`).

### Locales

```none
UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 18: ordinal not in range(128)
```

Set a UTF-8 locale:

```bash
export LANG=en_US.UTF-8
```

and, if it isn't installed,

```bash
sudo apt install locales
sudo locale-gen --no-purge --lang en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8 LANGUAGE
```

### Intel MKL

```none
Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so.1 or libmkl_def.so.1.
```

A known issue of conda's MKL; preload its libraries:

```bash
LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so altar-linear --config=...
```
