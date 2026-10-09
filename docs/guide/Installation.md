(installation)=
# Installation Guide

AlTar is built on the [pyre](https://github.com/pyre/pyre) framework, and both are installed
from source with [CMake](https://cmake.org), or with pyre's own build tool, {ref}`mm
<installation-mm>`. The recommended way is to build both into a conda environment, which also
supplies every library they need:

1. check the {ref}`requirements <installation-requirements>`;
2. create a {ref}`conda environment <installation-conda>` with the prerequisites;
3. {ref}`download <installation-downloads>` pyre and AlTar;
4. build and install {ref}`pyre <installation-pyre>`, then {ref}`AlTar <installation-altar>`;
5. {ref}`check <installation-check>` the installation.

The GPU (CUDA) support is optional: without a CUDA toolkit, both packages build their cpu parts
only, and AlTar runs every model that has a cpu implementation.

(installation-requirements)=
## Requirements

### Hardware and operating systems

- **CPU**: any 64-bit processor supported by your compiler.
- **GPU** (optional): an NVIDIA GPU supported by your CUDA toolkit. AlTar computes in single or
  double precision (`job.precision`, or `job.gpuprecision` for the GPU alone); consumer cards have
  far fewer double-precision units, so single precision is much faster on them, but some problems
  need double precision (see the cascaded example in {doc}`Kinematic`).
- **Operating systems**: Linux. Windows users can use the
  [Windows Subsystem for Linux](https://learn.microsoft.com/windows/wsl/) (WSL2), which also
  supports CUDA. macOS has not been tested with the current version.

The current version has been tested on Ubuntu 24.04 (under WSL2) with GCC 13.3, CUDA 13.2,
Python 3.13 and CMake 4.1, on an NVIDIA RTX 4060.

### Software

| package | version | notes |
|---|---|---|
| Python | ≥ 3.11 | with `numpy` and `h5py` |
| C++ compiler | C++23 | GCC ≥ 13, or a recent clang |
| CMake | ≥ 3.20 | and `make` |
| pybind11 | | for the python extension modules |
| HDF5 | ≥ 1.14.4 | data files |
| MPI | | optional: for running on several processes or nodes, e.g. Open MPI |
| CUDA toolkit | | optional: for GPU computations, with `cublas`, `curand` and `cusolver` |
| cuTile | | optional: `cuda-tile`, for the GPU kernels of the volcano models; with its compiler, `tileiras`, from a CUDA ≥ 13.2 toolkit or `pip install cuda-tile[tileiras]` |
| PyYAML | | optional: for `.yaml` configuration files |
| PostgreSQL client library | | optional: pyre's database support, not used by AlTar |

(installation-conda)=
## Prepare a conda environment

Install [Miniforge](https://github.com/conda-forge/miniforge) (or Miniconda/Anaconda) if you
don't have conda yet, then create an environment with the prerequisites from conda-forge:

```bash
conda create -n altar2 -c conda-forge python=3.13 numpy h5py hdf5 pybind11 pyyaml cmake make git
# optional, for MPI runs
conda install -n altar2 -c conda-forge openmpi
conda activate altar2
# for GPU support: pyre finds the GPUs with cuda-python
pip install cuda-python
# optional, for the GPU kernels of the volcano models
pip install cuda-tile
```

Without `cuda-python`, AlTar runs on the cpu, even with `job.gpus = 1`.

The C++ and CUDA compilers come from your system: a GCC 13 or newer on the `PATH`, and, for GPU
support, the CUDA toolkit's `nvcc` (e.g. `/usr/local/cuda/bin`). Check them with

```bash
g++ --version
nvcc --version   # for GPU support
```

If the system's GCC is older, e.g. on an older Linux distribution, install a newer one into the
environment, `conda install -n altar2 -c conda-forge gxx=14`, which then comes first on the `PATH`.

pyre and AlTar install their python packages under `$CONDA_PREFIX/packages`; make that a link to
the environment's `site-packages`, so that python finds them:

```bash
ln -sf "$(python -c 'import site; print(site.getsitepackages()[0])')" "$CONDA_PREFIX/packages"
```

(installation-downloads)=
## Download the sources

Choose a directory for the sources, e.g. `~/tools/src`, and clone pyre and AlTar into it:

```bash
mkdir -p ~/tools/src
cd ~/tools/src
git clone -b altar2 https://github.com/lijun99/pyre.git
git clone -b develop https://github.com/lijun99/altar.git
```

The `altar2` branch of pyre carries the pyre changes this version of AlTar needs, ahead of
their merge into pyre's own repository.

(installation-pyre)=
## Build and install pyre

```bash
cd ~/tools/src/pyre
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DWITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
    -DPYRE_BUILD_TESTING=OFF
cmake --build build -j 4
cmake --install build
```

pyre builds its CUDA support only when asked, with `-DWITH_CUDA=ON`; leave that option (and
`CMAKE_CUDA_ARCHITECTURES`) out for a cpu-only build. `-DPYRE_BUILD_TESTING=OFF` skips pyre's own
test suite, which takes a long time to compile.

(installation-altar)=
## Build and install AlTar

```bash
cd ~/tools/src/altar
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j 4
cmake --install build
```

AlTar enables its CUDA support by itself when it finds both a CUDA toolkit and a pyre built with
CUDA; otherwise it reports `CUDA Toolkit or Pyre cuda extension not found; set WITH_CUDA to OFF`
and builds the cpu parts only. Pass `-DWITH_CUDA=OFF` to skip the GPU parts on purpose.

The models (linear, seismic, ...) are built and installed along with the framework, together with
their command line applications, e.g. `altar-linear` and `slipmodel.plexus`.

```{note}
Compiling the CUDA sources takes a lot of memory. If the machine runs short, e.g. under WSL2, lower
the number of parallel jobs, `-j 2` or `-j 1`.
```

(installation-options)=
## CMake options

Both packages take the standard CMake options, among them

`CMAKE_INSTALL_PREFIX`
: where to install; with conda, `$CONDA_PREFIX`. For another location, e.g. `~/tools`, add its
  `bin` to `PATH`, its `lib` to `LD_LIBRARY_PATH`, and its `packages` to `PYTHONPATH`.

`CMAKE_PREFIX_PATH`
: where to look for the prerequisites; with conda, `$CONDA_PREFIX`.

`CMAKE_CUDA_ARCHITECTURES`
: the GPU architectures to compile for: `native` for the GPU(s) in the machine that builds, or
  compute capabilities such as `"80;89"` when building for other machines, e.g. the nodes of a
  cluster. See NVIDIA's [list of compute capabilities](https://developer.nvidia.com/cuda-gpus).

`CMAKE_BUILD_TYPE`
: `Release` (optimized), `Debug`, or `RelWithDebInfo`.

`WITH_CUDA`
: `ON`/`OFF`: build the GPU support (pyre: off unless asked; AlTar: on when available).

The compilers are picked from the `CXX` and `CUDACXX` environment variables, e.g.
`CXX=g++-13 CUDACXX=/usr/local/cuda/bin/nvcc cmake -S . -B build ...`. To see the full compile
commands, add `--verbose` to `cmake --build`.

(installation-mm)=
## Build with mm

Instead of CMake, pyre and AlTar can be built with [mm](https://github.com/aivazis/mm), pyre's own
build tool, which finds the sources itself and reads its description of the projects from their
`.mm` directories.

### Set up mm

Get mm, and make a shortcut for it, e.g. in `~/.bashrc`:

```bash
git clone https://github.com/aivazis/mm.git ~/tools/src/mm
alias mm='python3 ${HOME}/tools/src/mm/mm'
```

Tell mm to build into the active conda environment, in `~/.config/pyre/mm.yaml`:

```yaml
mm:
  # optimized, shared libraries
  target: "opt, shared"
  # install into the active conda environment, and find the prerequisites there
  mode: conda
  pkgdb: conda
  # the compilers
  compilers: "gcc, python/python3"
  # the intermediate build products, one directory per environment
  bldroot: "{pyre.environ.HOME}/tmp/builds/mm/{pyre.environ.CONDA_DEFAULT_ENV}"
```

and where to find what the environment doesn't provide, in `~/.config/mm/config.mm`: pyre, once
it is installed in the environment, and, for GPU support, the CUDA toolkit:

```make
# the conda environment
sys.prefix := ${CONDA_PREFIX}
# pyre
pyre.dir := $(sys.prefix)
# the CUDA toolkit
cuda.dir := /usr/local/cuda
```

### Build

With the environment active, build pyre, then AlTar, each from its source directory:

```bash
cd ~/tools/src/pyre
mm --slots=4
cd ~/tools/src/altar
mm --slots=4
```

`--slots` sets the number of parallel jobs; as with CMake, lower it if the machine runs short of
memory. mm installs everything, the python packages included, straight into the environment, so no
`packages` link is needed. It builds the GPU parts of AlTar, `altar-cuda` and `seismic-cuda`, only
when it finds CUDA, through `cuda.dir`.

The projects AlTar builds, and where their sources are, are listed in `.mm/projects.mm` and in a
file per project, e.g. `.mm/altar.mm` for the framework and `.mm/linear.mm` for the linear model.
Use one of mm and CMake in an environment. They install to the same places, but CMake installs the
python sources and mm only their compiled modules, which python ignores when a source is present:
to switch from CMake to mm, remove the python packages CMake installed first, e.g.
`$CONDA_PREFIX/lib/python3.13/site-packages/altar`.

(installation-check)=
## Check the installation

```bash
python -c "import altar; print(altar.version())"
```

prints the version, and the linear model example runs on the cpu with

```bash
cd ~/tools/src/altar/models/linear/examples
altar-linear --config=linear_catmip.pfg
```

and, with GPU support, on the GPU with

```bash
altar-linear --config=linear_catmip.pfg --job.gpus=1
```

Each run anneals from β = 0 to β = 1, in about 20 steps, and prints the posterior mean and
standard deviation of each parameter; a few minutes on the cpu, well under a minute on a GPU.

The tests check an installation from the source tree, in two suites, the cpu one and the GPU
one:

```bash
cd ~/tools/src/altar
python tests/run.py                # the cpu suite, a few minutes
python tests/run.py --gpu          # the GPU suite: the cuda tests, examples, posteriors
python tests/run.py --list         # the tests, and what each needs
python tests/run.py linear         # the tests whose names contain "linear"
```

Each suite runs the framework and model checks, short runs of the examples, and the exact
posterior test of the linear model. Tests that need something optional are skipped without it:
jax, cuTile for the volcano models on the GPU, two GPUs for the MPI one, and the 9patch inputs
of the seismic examples, which aren't in the repository (`--seismic-data`, or
`$ALTAR_SEISMIC_DATA`). A failed test leaves its log in `test-logs`.

(installation-mpi)=
## MPI

pyre finds MPI through its own configuration: if more than one MPI is installed, e.g. the
system's and the conda environment's, tell it which one to use in `~/.config/pyre/mpi.pfg`, e.g.
for the Open MPI in the conda environment

```ini
; pick an MPI implementation
mpi.shells.mpirun:
    mpi = openmpi#mpi_conda

; the Open MPI in the conda environment
pyre.externals.mpi.openmpi # mpi_conda:
    version = 5.0
    launcher = /path/to/conda/envs/altar2/bin/mpirun
    prefix = /path/to/conda/envs/altar2
    bindir = {mpi_conda.prefix}/bin
    incdir = {mpi_conda.prefix}/include
    libdir = {mpi_conda.prefix}/lib
```

with the paths of your environment (`echo $CONDA_PREFIX`). Give the `launcher` as a full path:
pyre otherwise runs the first `mpirun` on the `PATH`, which may belong to another MPI.

An MPI run then only needs the number of tasks and the shell, e.g.

```bash
altar-linear --config=linear_catmip.pfg --job.tasks=2 --shell=mpi.shells.mpirun
```

See the {doc}`Manual` for running on clusters.
