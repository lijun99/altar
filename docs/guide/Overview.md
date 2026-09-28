(overview)=
# Overview

AlTar samples the posterior distribution of inverse problems with {ref}`CATMIP <catmip>` and other
Markov chain Monte Carlo methods. It is built on the component architecture and the job management
of the [pyre](https://github.com/pyre/pyre) framework, so that a simulation is assembled, and
changed, from a configuration file rather than by programming.

An AlTar application has three main components:

- the **model**, which evaluates the forward problem and so the data likelihood, along with the
  prior distributions of its parameters;
- the **controller**, which samples the posterior: the annealing schedule, the sampler that moves
  the chains (Metropolis, HMC, MALA, ...), and the archiver that saves the results;
- the **job**, which sets the size of the simulation, the number of chains and of steps, and how it
  is deployed: cpu or GPU, one or several processes, a workstation or a cluster.

Each component can be configured: switched to another implementation, turned on or off, or given
different parameter values. A simulation is run with a single command,

```bash
anAlTarApp --config=anAlTarApp.pfg
```

where `anAlTarApp` is the AlTar application for an inverse problem, e.g. `altar-linear` for the
linear model, and `anAlTarApp.pfg` is the configuration file with the settings of the simulation.
Each model comes with examples that serve as templates for your own simulations.

This guide is organized as follows:

- {doc}`QuickStart`: running a simulation, with the linear model;
- {doc}`Pyre`: pyre components and the `.pfg` configuration files;
- {doc}`AlTarFramework`: configuring the controller, its samplers and archivers, and the job;
- {doc}`Priors`: the prior distributions, and their reparameterization for gradient-based
  samplers;
- {doc}`Models`: preparing the data and running the geophysical models that come with AlTar:
  - {doc}`Static`: static slip inversion of earthquake sources;
  - {doc}`StaticCp`: accounting for the uncertainty of the forward model, $C_p$;
  - {doc}`Kinematic`: kinematic slip inversion, solved together with the static one, in a cascade.

To develop a model for your own inverse problem, see the {doc}`Programming`.
