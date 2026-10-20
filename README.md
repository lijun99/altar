# AlTar 2.0

## About AlTar
AlTar(Al-Tar) is a software package to solve inverse problems with Bayesian inference. It is named after the Spanish-French physicist [Albert Tarantola](https://en.wikipedia.org/wiki/Albert_Tarantola), for his pioneering work on the Bayesian perspective of inverse problems.

AlTar samples the posterior distribution of a model with Markov chain Monte Carlo methods: the
[Cascading Adaptive Tempered Metropolis In Parallel (CATMIP) algorithm](https://thesis.library.caltech.edu/5918/),
designed for high-dimensional problems on parallel computers, and, since AlTar 2, gradient-based
samplers (HMC, MALA, SGLD). The same configuration runs on the cpu or on NVIDIA GPUs, in one
process or many under MPI, on a workstation or a SLURM cluster.

See [WHATSNEW.md](https://github.com/lijun99/altar/blob/develop/WHATSNEW.md) for what changed in AlTar 2, and what changes for existing runs.

## What's in the package

**Sampling algorithms**:

| algorithm | controller |
|---|---|
| CATMIP: tempered annealing from the prior to the posterior, with Metropolis moves | `catmip` |
| CATMIP with Hamiltonian Monte Carlo (HMC) moves | `catmip_hmc` |
| CATMIP with Metropolis-adjusted Langevin (MALA) moves | `catmip_mala` |
| Cross-fade CATMIP, from the conjugate posterior of a linear model (Minson, 2024) | `cf_catmip` |
| Adaptive Metropolis, directly at the posterior | `mcmc` |
| Hamiltonian Monte Carlo, directly at the posterior | `hmc` |
| Metropolis-adjusted Langevin, directly at the posterior | `mala` |
| Stochastic gradient Langevin dynamics (SGLD) | `langevin` |

Each is selected with `controller = altar.bayesian.<controller>`.

**Geophysical models**:

| model | package |
|---|---|
| Linear inverse problems, d = G θ | `linear` |
| Static slip inversion of earthquakes, with model uncertainties (C_p) | `seismic` |
| Kinematic slip inversion, on rectangular or triangulated faults, alone or jointly with the static one | `seismic` |
| Volcanic sources: Mogi point source, compound dislocation model (CDM), Reverso | `mogi`, `cdm`, `reverso` |
| Sequences of earthquakes and aseismic slip (SEAS), *in development* | `seas` |

**Adding a model** takes little code: it only needs a forward model, written in numpy. GPU
kernels can be written with cuTile, NVRTC or CUDA C++; see the [Programming Guide](https://altar2.readthedocs.io/en/latest/guide/Programming.html).

## Installation

AlTar is built on [pyre](https://github.com/pyre/pyre), which it carries as a git submodule; pip
builds the two together and installs them into the current Python environment:

```bash
git clone --recursive -b develop https://github.com/lijun99/altar.git
cd altar
pip install ".[cuda]"     # or "." for the cpu only
```

The build needs a C++23 compiler (e.g. GCC ≥ 13) and CMake, and, for GPU support, the CUDA
toolkit. See the [Installation Guide](https://altar2.readthedocs.io/en/latest/guide/Installation.html)
for the requirements, a ready conda environment, the build options (GPU architectures, cpu only,
incremental rebuilds), and building pyre and AlTar separately with CMake.

## Guides

Readthedocs ([html](https://altar2.readthedocs.io/en/latest/) | [pdf](https://altar2.readthedocs.io/_/downloads/en/latest/pdf/) | [epub](https://altar2.readthedocs.io/_/downloads/en/latest/epub/)):

- [Installation Guide](https://altar2.readthedocs.io/en/latest/guide/Installation.html)
- [Quick Start](https://altar2.readthedocs.io/en/latest/guide/QuickStart.html)
- [User Guide](https://altar2.readthedocs.io/en/latest/guide/Manual.html)
- [Programming Guide](https://altar2.readthedocs.io/en/latest/guide/Programming.html)
- [API Reference](https://altar2.readthedocs.io/en/latest/api/altar/index.html)

The sources of the documentation are in `docs/`. The documentation of the previous, CUDA
version of AlTar 2.0 remains at [altar.readthedocs.io](https://altar.readthedocs.io).

## Tutorials
Jupyter notebooks, in their own repository, [altar2-tutorials](https://github.com/lijun99/altar2-tutorials).

## Support

- [Issues @ github](https://github.com/lijun99/altar/issues)
- [Common Issues](https://altar2.readthedocs.io/en/latest/guide/Issues.html)
- [Slack Discussion Group](https://altar-group.slack.com) (currently only for developers)

## Copyright

```text
    Copyright (c) 2013-present ParaSim Inc.
    Copyright (c) 2010-present California Institute of Technology
    All Rights Reserved

    This software is subject to the provisions of its LICENSE, a copy of
    which should accompany all distributions, in both source and binary
    form. If you received this software without a copy of the LICENSE,
    please contact the author at michael.aivazis@para-sim.com.

    THIS SOFTWARE IS PROVIDED "AS IS" AND ANY AND ALL EXPRESS OR IMPLIED
    WARRANTIES ARE DISCLAIMED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF TITLE, MERCHANTABILITY, AGAINST INFRINGEMENT, AND FITNESS
    FOR A PARTICULAR PURPOSE.
```
