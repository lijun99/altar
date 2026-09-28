(pyre-framework)=
# Pyre Basics

AlTar is built on the [pyre](https://github.com/pyre/pyre) framework. This page introduces the
parts of pyre an AlTar user meets: protocols and components, and the `.pfg` configuration files.

## Protocols and components

Pyre extends python classes into *components*, whose attributes can be configured, and whose
implementation can be swapped, at run time.

A *protocol* declares a role, e.g. a probability distribution, and the behaviors any
implementation of it must provide; `@altar.provides` marks them. The protocol that AlTar's
distributions implement:

```{literalinclude} ../../altar/altar/distributions/Distribution.py
:language: python
:pyobject: Distribution
```

A *component* implements a protocol: it declares the protocols it implements, its configurable
*traits*, and the behaviors, marked with `@altar.export`. The uniform distribution, for example:

```{literalinclude} ../../altar/altar/distributions/Uniform.py
:language: python
:pyobject: Uniform
```

Its traits are

- *properties*, configurable values of basic types, such as `support`, an array, and
  `reparameterize`, a boolean;
- *facilities*, configurable components, such as `transform`, any component that implements the
  transform protocol;
- and, like any python class, ordinary attributes, which are not configurable.

The uniform distribution declares its traits and nothing else: its numerics live in plain python
classes, one for the cpu and one for the GPU, and it picks one of them when it is initialized (see
the {doc}`Programming`). Its other behaviors, and the `parameters` and `offset` traits, come from
its base class.

Components are the building blocks of an AlTar application. A parameter set, for example, has a
prior distribution, which can be any component that implements the `Distribution` protocol:

```{literalinclude} ../../altar/altar/models/Contiguous.py
:language: python
:pyobject: Contiguous
```

so that `prior = gaussian` in a configuration file selects the gaussian distribution. When a
facility isn't configured, its protocol supplies a default implementation (`pyre_default`).

```{note}
Components are configured and instantiated by pyre as part of an application. Created by hand, in
a python shell, they don't pick up any configuration.
```

(pyre-config-format)=
## The configuration files (`.pfg`)

Properties and components can be configured on the command line, or, more conveniently, in a
configuration file. Pyre reads `.pfg` files (a format similar to YAML), `.cfg` files (an INI-style
format used by AlTar 1.1) and `.pml` files (XML). We recommend `.pfg`; see {doc}`QuickStart` for an
example.

The rules of the `.pfg` format are:

- **Indentation** gives the hierarchy; tab characters are not allowed.
- **Comments** start with `;`.
- **Paths**: a setting can be given by indentation, by its full path, or by a partial path under
  an indented parent. These three are equivalent:

  ```none
  ; all by indentation
  linear:
      job:
          tasks = 1
          gpus = 0
          chains = 2**10

  ; all by full path
  linear.job.tasks = 1
  linear.job.gpus = 0
  linear.job.chains = 2**10

  ; partial paths under an indented parent
  linear:
      job.tasks = 1
      job.gpus = 0
      job.chains = 2**10
  ```

- **Choosing a component**: a facility is set by the name of an implementation, e.g.
  `prior = gaussian` for one of AlTar's own distributions, or by its full name, e.g.
  `prior = altar.models.seismic.moment` for one that comes with a model, or
  `controller = altar.bayesian.catmip`. Its own settings then follow, indented below it:

  ```none
  prior = altar.models.seismic.moment
  prior:
      support = (-0.5, 20)
      Mw_mean = 7.3
  ```

- **Values** need no quotation marks, even strings and paths. Numbers may be expressions, e.g.
  `2**10`, and values may refer to other settings in braces, e.g.
  `count = {linear.model.parameters}`.
- **Defaults**: anything left out keeps the default of its component.

On the command line, the same settings take the form `--path=value`, e.g.
`--job.chains=2**10` or `--controller.sampler=altar.bayesian.samplers.hmc`; they override the
configuration file. Pyre also reads user-wide settings from `~/.config/pyre`, e.g. the MPI setup
in `~/.config/pyre/mpi.pfg` (see {doc}`Installation`).
