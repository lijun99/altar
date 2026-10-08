(volcano-models)=
# Volcano Deformation Models

Three models of the surface deformation of volcanoes, from the simplest to the richest:

| model | source | parameters | data |
|---|---|---|---|
| `altar.models.mogi` | a point pressure source | 4, and the offsets | displacements along a line of sight, e.g. InSAR |
| `altar.models.cdm` | a compound dislocation model, an opening box of any shape and orientation | 10, and the offsets | displacements along a line of sight |
| `altar.models.reverso` | two magma chambers connected by a conduit, over time | 6 | east, north and up displacements over time, e.g. GPS |

The forward models are nonlinear; the models are sampled with CATMIP, on the cpu, in C++, or on
the GPU, with [cuTile](https://docs.nvidia.com/cuda/cutile-python) kernels (see
{doc}`Programming`). Each comes with a generator of synthetic data, an example configuration, and
an application, `altar-mogi`, `altar-cdm` and `altar-reverso`.

## The models

### Mogi

The Mogi [1958] model is a point source of pressure at depth $d$ under $(x_0, y_0)$, in an elastic
half space with Poisson's ratio $\nu$. A volume change $\Delta V$ moves the surface point
$(x, y)$ by

$$
\mathbf u = \frac{(1-\nu)\,\Delta V}{\pi R^3} \left(x - x_0,\; y - y_0,\; d\right),
\qquad R^2 = (x-x_0)^2 + (y-y_0)^2 + d^2,
$$

seen along the line of sight (LOS) of each observation. Its parameter sets are

`location`
: $(x_0, y_0)$, count 2.

`depth`
: $d$, count 1.

`source`
: $\Delta V$ in m³, negative for a deflation; or $\log_{10}\Delta V$ with `log10_dV = true`.

`offsets` (optional)
: one shift per dataset, subtracted from the predicted displacements of its observations, e.g.
  for the arbitrary reference of an interferogram.

### CDM

The compound dislocation model of Nikkhoo et al. [2017] is three mutually orthogonal rectangular
dislocations with a common opening, centered at depth $d$ under $(x_0, y_0)$: an ellipsoid-like
source of any shape, from a sill to a dike to a pipe, and any orientation. AlTar's implementation
follows Nikkhoo's `CDM.m`, and reproduces its displacements (`models/cdm/tests/libcdm.py`). Its
parameter sets are

`location`, `depth`
: as for Mogi.

`opening`
: the opening of the dislocations, count 1, in the units of the coordinates.

`a`
: the semi-axes $(a_x, a_y, a_z)$ before rotation, count 3; or `aX`, `aY`, `aZ`, count 1 each, for
  priors that differ by axis.

`omega`
: the rotations $(\omega_x, \omega_y, \omega_z)$ about the x, y and z axes, in degrees, count 3; or
  `omegaX`, `omegaY`, `omegaZ`.

`offsets` (optional)
: as for Mogi.

The half space solution needs the whole source below the surface: the samples with a vertex above
it are rejected, as if outside the support of their priors.

### Reverso

The Reverso et al. [2014] model is a deep magma chamber, fed at a constant rate $Q_{in}$, that
feeds a shallow one through a cylindrical conduit of radius $a_c$. Starting from no overpressure,
the overpressures of the two chambers follow in closed form, with a characteristic time

$$
\tau = \frac{8 \mu H_c \gamma_s \gamma_d k\, a_s^3}{G a_c^4 \gamma_r},
\qquad k = \left(\frac{a_d}{a_s}\right)^3, \quad H_c = H_d - H_s, \quad
\gamma_r = \gamma_s + \gamma_d k,
$$

and each chamber, a sill or a sphere (`shallow`, `deep`), moves the surface as a point source
under the origin. Its parameter sets, count 1 each, are `Qin`, in m³/s, `H_s`, `a_s` and `H_d`,
`a_d`, the depths and radii of the shallow and deep chambers, and `a_c`, in m. The samples whose
deep chamber isn't below the shallow one are rejected. The medium is set by `G`, `v`, `mu` (the
magma viscosity), `drho` (the density of the rock less that of the magma) and `g`.

## Input

Each model reads three files, in the directory `model.case`:

the data
: the observed displacements, a vector, read by `dataobs` (`dataobs.data_file`): for Mogi and CDM
  the LOS displacements, for Reverso the east, north and up displacements of each observation, in
  that order.

the data covariance
: an observations × observations matrix (`dataobs.cd_file`), or a constant variance
  (`dataobs.cd_std`).

the geometry
: a csv file (`model.geometry`, `geometry.csv` by default), one row per observation, in the order
  of the data. For Mogi and CDM, the columns are
  ```none
  oid,x,y,theta,phi
  ```
  the dataset of the observation, for its offset; its location; and its LOS, the unit vector from
  the ground to the observing craft, by its incidence angle `theta` from the vertical and its
  azimuth `phi` counterclockwise from east, both in radians,
  $\left(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta\right)$ in (east, north, up). For
  Reverso, the columns are
  ```none
  oid,t,x,y
  ```
  the time of the observation, in seconds since the start of the inflow, and its location, relative
  to the chambers.

`dataobs.observations` is the length of the data: the number of rows of the geometry for Mogi and
CDM, three times that for Reverso.

## Running the examples

Each model generates its synthetic data, in `models/<model>/examples/synthetic`, from a source
given on the command line, e.g. for Mogi

```bash
cd ~/tools/src/altar/models/mogi/examples/synthetic
python3 mogi.py                 # or, e.g., python3 mogi.py --dV=-5e6 --noise=true
cd ..
altar-mogi --config=mogi.pfg    # on the cpu
altar-mogi --config=mogi.pfg --job.gpus=1
```

and likewise `cdm.py` and `altar-cdm --config=cdm.pfg`, `reverso.py` and
`altar-reverso --config=reverso.pfg`. The Mogi data are seen by one track, the CDM data by an
ascending and a descending track, and the Reverso data by GPS stations over a year. Each run
anneals to β = 1 and prints the posterior mean and standard deviation of each parameter, which
recover the source of the synthetic data; on a laptop GPU, in seconds for Mogi and Reverso, about
a minute and a half for CDM.

The Mogi example:

```{literalinclude} ../../models/mogi/examples/mogi.pfg
:language: none
:caption: mogi.pfg
:lines: 13-
```

```{note}
The parameters of these models differ by orders of magnitude, e.g. a volume change of 10⁷ m³ and
offsets of centimeters. The conditioning of the proposal covariance raises its eigenvalues to a
fraction of the largest one, which, at these scales, swamps the small parameters: the examples turn
it off, with `controller.scheduler.check_positive_definiteness = False` and
`controller.sampler.proposal.check_positive_definiteness = False`.
```

```{note}
The CDM's ten parameters trade off against each other, e.g. its opening against its size; it
needs many chains, 4096 in the example, to find the posterior.
```

## Precision on the GPU

The Mogi kernel computes in the precision of the job, `job.gpuprecision`. The CDM and Reverso
kernels always compute in double precision, and store in the precision of the job: in single
precision, the CDM's angular dislocations cancel to errors of up to a fifth of the signal. On a GPU
with slow double precision arithmetic, such as a consumer card, CDM runs no faster in single
precision.

## References

- K. Mogi (1958), Relations between the eruptions of various volcanoes and the deformations of the
  ground surfaces around them, *Bull. Earthq. Res. Inst.*, 36, 99–134.
- M. Nikkhoo, T. R. Walter, P. R. Lundgren, P. Prats-Iraola (2017), Compound dislocation models
  (CDMs) for volcano deformation analyses, *Geophys. J. Int.*, 208(2), 877–894.
- T. Reverso, J. Vandemeulebrouck, F. Jouanne, V. Pinel, T. Villemin, E. Sturkell, P. Bascou (2014),
  A two-magma chamber model as a source of deformation at Grímsvötn Volcano, Iceland, *J. Geophys.
  Res. Solid Earth*, 119, 4666–4683.
